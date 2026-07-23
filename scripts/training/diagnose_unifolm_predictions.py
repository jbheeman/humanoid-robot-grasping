#!/usr/bin/env python3
"""Diagnose UniFoLM outputs against trivial baselines and visual perturbations."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src/object_tracking").is_dir():
        sys.path.insert(0, str(_parent / "src"))
        break

from object_tracking.unifolm_relative_actions import (  # noqa: E402
    RELATIVE_POSE23_V1,
    reconstruct_anchored_pose23,
)

RIGHT_XYZ = slice(9, 12)


def denormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
    return np.where(
        mask,
        (values + 1.0) * 0.5 * (high - low) + low,
        values,
    )


def summarize_errors(
    errors: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    valid = (
        np.ones(values.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != values.shape or not np.any(valid):
        raise ValueError("error validity mask must match errors and contain valid targets")
    final = np.asarray(
        [row[np.flatnonzero(mask)[-1]] for row, mask in zip(values, valid)]
    )
    horizon_counts = np.sum(valid, axis=0)
    horizon_means = np.divide(
        np.sum(np.where(valid, values, 0.0), axis=0),
        horizon_counts,
        out=np.full(values.shape[1], np.nan),
        where=horizon_counts > 0,
    )
    return {
        "ade_m": float(np.mean(values[valid])),
        "fde_m": float(np.mean(final)),
        "median_m": float(np.median(values[valid])),
        "p95_m": float(np.percentile(values[valid], 95)),
        "per_horizon_mean_m": [
            float(value) if count else None
            for value, count in zip(horizon_means, horizon_counts)
        ],
        "per_horizon_valid_count": horizon_counts.tolist(),
    }


def collate_one(example: dict, pad_token_id: int) -> dict:
    import torch

    input_ids = example["input_ids"].squeeze(0)
    output = {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": input_ids.ne(pad_token_id).unsqueeze(0),
        "action": torch.as_tensor(np.stack([example["actions"]])),
        "action_valid_mask": torch.as_tensor(
            np.stack([example["action_valid_mask"]])
        ),
    }
    proprio = np.stack([example["proprio"]])
    if proprio.ndim == 3 and proprio.shape[1] == 1:
        proprio = proprio[:, 0, :]
    output["state"] = torch.as_tensor(proprio)
    for key in ("pixel_values", "image_grid_thw"):
        if key in example:
            output[key] = example[key]
    return output


def latest_right_xyz(state: np.ndarray) -> np.ndarray:
    """Return the newest right-hand XYZ from [B,D] or causal [B,T,D] state."""
    values = np.asarray(state)
    if values.ndim == 2 and values.shape[-1] >= RIGHT_XYZ.stop:
        return values[:, RIGHT_XYZ]
    if values.ndim == 3 and values.shape[-1] >= RIGHT_XYZ.stop:
        return values[:, -1, RIGHT_XYZ]
    raise ValueError(f"unexpected proprio state shape: {values.shape}")


def latest_pose23(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state)
    if values.ndim == 2 and values.shape[-1] == 23:
        return values
    if values.ndim == 3 and values.shape[-1] == 23:
        return values[:, -1, :]
    raise ValueError(f"unexpected proprio state shape: {values.shape}")


def to_device(batch: dict, device: Any) -> dict:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_action_checkpoint(model: Any, checkpoint: Path) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "action_model."
    action_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not action_state:
        raise ValueError(f"{checkpoint} contains no action_model tensors")
    model.action_model.load_state_dict(action_state, strict=True)
    del state, action_state
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=48)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    os.environ["G1_PLUSH_SHARED_STATS"] = str(args.stats.resolve())
    repo_root = args.config.parents[2]
    sys.path.insert(0, str((repo_root / "unifolm-vla" / "src").resolve()))

    import tensorflow as tf
    import torch
    from omegaconf import OmegaConf
    from unifolm_vla.model.framework import build_framework
    from unifolm_vla.rlds_dataloader.datasets.datasets import (
        RLDSBatchTransform,
        RLDSDataset,
    )

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    window_size = int(cfg.datasets.vla_data.get("window_size", 1))
    observation_stride = int(cfg.datasets.vla_data.get("observation_stride", 1))
    model = build_framework(cfg)
    load_action_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.action_model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    processor = model.qwen_vl_interface.processor
    transform = RLDSBatchTransform(
        processor=processor,
        use_wrist_image=False,
        use_proprio=True,
    )
    stats = json.loads(args.stats.read_text())
    action_stats = stats["action"]
    proprio_stats = stats["proprio"]
    action_representation = stats.get("representation", {}).get(
        "version", "absolute_pose23"
    )
    if action_representation not in ("absolute_pose23", RELATIVE_POSE23_V1):
        raise ValueError(f"unsupported action representation: {action_representation}")
    per_horizon_mean = None
    if action_representation == RELATIVE_POSE23_V1:
        horizon_stats = stats["provenance"]["per_horizon"]
        per_horizon_mean = np.asarray(
            [horizon_stats[str(index)]["action"]["mean"] for index in range(25)],
            dtype=np.float32,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "samples_per_source": args.samples_per_source,
        "seed": args.seed,
        "window_size": window_size,
        "observation_stride": observation_stride,
        "action_representation": action_representation,
        "history_span_s_at_30hz": (window_size - 1)
        * observation_stride
        / 30.0,
        "sources": {},
        "physical_robot_authorized": False,
    }

    for source in ("g1_plush_touch_real", "g1_plush_touch_sim"):
        dataset = RLDSDataset(
            args.data_root,
            source,
            transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=max(args.samples_per_source, 64),
            train=False,
            split_override=args.split,
            image_aug=False,
            window_size=window_size,
            observation_stride=observation_stride,
            action_representation=action_representation,
            relative_action_statistics=(
                str(args.stats)
                if action_representation == RELATIVE_POSE23_V1
                else None
            ),
        )
        iterator = iter(dataset)
        batches = [
            collate_one(next(iterator), processor.tokenizer.pad_token_id)
            for _ in range(args.samples_per_source)
        ]

        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        current_predictions: list[np.ndarray] = []
        mean_predictions: list[np.ndarray] = []
        shuffled_predictions: list[np.ndarray] = []
        occluded_predictions: list[np.ndarray] = []
        normalized_saturation: list[float] = []
        valid_masks: list[np.ndarray] = []

        for index, cpu_batch in enumerate(batches):
            batch = to_device(cpu_batch, device)
            target_normalized = batch["action"].float().cpu().numpy()
            target = denormalize(target_normalized, action_stats)
            state_normalized = batch["state"].float().cpu().numpy()
            state = denormalize(state_normalized, proprio_stats)
            anchor = latest_pose23(state)
            if action_representation == RELATIVE_POSE23_V1:
                target = reconstruct_anchored_pose23(anchor[:, None, :], target)
            current = np.repeat(
                latest_right_xyz(state)[:, None, :],
                target.shape[1],
                axis=1,
            )
            if action_representation == RELATIVE_POSE23_V1:
                assert per_horizon_mean is not None
                mean_absolute = reconstruct_anchored_pose23(
                    anchor[:, None, :],
                    per_horizon_mean[None, :, :],
                )
                mean_prediction = mean_absolute[:, :, RIGHT_XYZ]
            else:
                mean_action = np.asarray(action_stats["mean"], dtype=np.float32)
                mean_prediction = np.broadcast_to(
                    mean_action[None, None, RIGHT_XYZ],
                    current.shape,
                )

            torch.manual_seed(args.seed + index)
            with torch.inference_mode():
                original_normalized = np.asarray(
                    model.predict_action(qwen_inputs=batch)["normalized_actions"],
                    dtype=np.float32,
                )

            shuffled = dict(batch)
            donor = batches[(index + 1) % len(batches)]
            for key in ("pixel_values", "image_grid_thw"):
                if key in donor:
                    shuffled[key] = donor[key].to(device)
            torch.manual_seed(args.seed + index)
            with torch.inference_mode():
                shuffled_normalized = np.asarray(
                    model.predict_action(qwen_inputs=shuffled)["normalized_actions"],
                    dtype=np.float32,
                )

            occluded = dict(batch)
            if "pixel_values" in occluded:
                occluded["pixel_values"] = torch.zeros_like(occluded["pixel_values"])
            torch.manual_seed(args.seed + index)
            with torch.inference_mode():
                occluded_normalized = np.asarray(
                    model.predict_action(qwen_inputs=occluded)["normalized_actions"],
                    dtype=np.float32,
                )

            targets.append(target[0, :, RIGHT_XYZ])
            valid_masks.append(
                batch["action_valid_mask"].bool().cpu().numpy()[0]
            )
            original = denormalize(original_normalized, action_stats)
            shuffled_action = denormalize(shuffled_normalized, action_stats)
            occluded_action = denormalize(occluded_normalized, action_stats)
            if action_representation == RELATIVE_POSE23_V1:
                original = reconstruct_anchored_pose23(anchor[:, None, :], original)
                shuffled_action = reconstruct_anchored_pose23(
                    anchor[:, None, :], shuffled_action
                )
                occluded_action = reconstruct_anchored_pose23(
                    anchor[:, None, :], occluded_action
                )
            predictions.append(original[0, :, RIGHT_XYZ])
            current_predictions.append(current[0])
            mean_predictions.append(mean_prediction[0])
            shuffled_predictions.append(shuffled_action[0, :, RIGHT_XYZ])
            occluded_predictions.append(occluded_action[0, :, RIGHT_XYZ])
            normalized_saturation.append(
                float(np.mean(np.abs(original_normalized) >= 0.999))
            )

        target_values = np.stack(targets)
        prediction_values = np.stack(predictions)
        current_values = np.stack(current_predictions)
        mean_values = np.stack(mean_predictions)
        shuffled_values = np.stack(shuffled_predictions)
        occluded_values = np.stack(occluded_predictions)
        valid = np.stack(valid_masks)
        model_errors = np.linalg.norm(prediction_values - target_values, axis=-1)
        current_errors = np.linalg.norm(current_values - target_values, axis=-1)
        mean_errors = np.linalg.norm(mean_values - target_values, axis=-1)
        signed_bias = prediction_values - target_values
        image_shuffle_displacement = np.linalg.norm(
            shuffled_values - prediction_values, axis=-1
        )
        occlusion_displacement = np.linalg.norm(
            occluded_values - prediction_values, axis=-1
        )
        initial_state = current_values[:, 0, :]
        predicted_displacement = np.linalg.norm(
            prediction_values - initial_state[:, None, :],
            axis=-1,
        )
        target_displacement = np.linalg.norm(
            target_values - initial_state[:, None, :],
            axis=-1,
        )
        mean_predicted_displacement = float(np.mean(predicted_displacement[valid]))
        mean_target_displacement = float(np.mean(target_displacement[valid]))
        displacement_ratio = mean_predicted_displacement / max(
            mean_target_displacement,
            1e-8,
        )

        model_summary = summarize_errors(model_errors, valid)
        baseline_summary = summarize_errors(current_errors, valid)
        mean_summary = summarize_errors(mean_errors, valid)
        best_baseline_ade = min(
            baseline_summary["ade_m"],
            mean_summary["ade_m"],
        )
        improvement = (best_baseline_ade - model_summary["ade_m"]) / best_baseline_ade
        per_sample_model_error = np.asarray(
            [np.mean(row[mask]) for row, mask in zip(model_errors, valid)]
        )
        worst = np.argsort(per_sample_model_error)[::-1][:8]
        signed_bias_valid = signed_bias[valid]
        prediction_valid = prediction_values[valid]
        target_valid = target_values[valid]
        shuffle_valid = image_shuffle_displacement[valid]
        occlusion_valid = occlusion_displacement[valid]
        report["sources"][source] = {
            "model": model_summary,
            "current_pose_baseline": baseline_summary,
            "mean_action_baseline": mean_summary,
            "best_baseline_ade_m": float(best_baseline_ade),
            "relative_improvement_over_best_baseline": float(improvement),
            "beats_best_baseline_by_10_percent": bool(improvement >= 0.10),
            "right_xyz_signed_bias_m": np.mean(signed_bias_valid, axis=0).tolist(),
            "right_xyz_prediction_std_m": np.std(
                prediction_valid, axis=0
            ).tolist(),
            "right_xyz_target_std_m": np.std(target_valid, axis=0).tolist(),
            "predicted_displacement_from_current_ade_m": mean_predicted_displacement,
            "target_displacement_from_current_ade_m": mean_target_displacement,
            "predicted_to_target_displacement_ratio": float(displacement_ratio),
            "image_shuffle_prediction_change_m": {
                "mean": float(np.mean(shuffle_valid)),
                "p95": float(np.percentile(shuffle_valid, 95)),
            },
            "image_occlusion_prediction_change_m": {
                "mean": float(np.mean(occlusion_valid)),
                "p95": float(np.percentile(occlusion_valid, 95)),
            },
            "normalized_output_saturation_fraction": float(
                np.mean(normalized_saturation)
            ),
            "diagnosis_flags": {
                "visually_conditioned": bool(
                    np.mean(shuffle_valid) >= 0.01
                    and np.mean(occlusion_valid) >= 0.01
                ),
                "action_magnitude_over_2x_target": bool(displacement_ratio >= 2.0),
                "normalized_output_saturation_over_5_percent": bool(
                    np.mean(normalized_saturation) >= 0.05
                ),
                "right_xyz_bias_over_3cm": bool(
                    np.linalg.norm(np.mean(signed_bias_valid, axis=0)) >= 0.03
                ),
            },
            "worst_sample_indices": worst.tolist(),
            "worst_sample_ade_m": per_sample_model_error[worst].tolist(),
        }

    failures = [
        source
        for source, metrics in report["sources"].items()
        if not metrics["beats_best_baseline_by_10_percent"]
    ]
    report["gate"] = {
        "passed": not failures,
        "criterion": (
            "model right-XYZ ADE is at least 10% below both current-pose and "
            "train-mean action baselines"
        ),
        "failed_sources": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        "PREDICTION_DIAGNOSTIC_PASS" if not failures else "PREDICTION_DIAGNOSTIC_FAIL",
        f"failed_sources={failures}",
        f"output={args.output}",
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
