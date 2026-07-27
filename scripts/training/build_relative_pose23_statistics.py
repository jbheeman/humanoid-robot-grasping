#!/usr/bin/env python3
"""Build train-only anchored-relative Pose23 statistics from canonical HDF5.

Only valid (non-terminal-padding) horizon targets contribute to the action
statistics. Absolute proprio statistics are copied byte-for-value from the
existing train-only shared statistics so the input contract does not drift.
This tool is offline and imports no ROS, Unitree SDK, or simulator modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from object_tracking.unifolm_relative_actions import (
    RELATIVE_POSE23_V1,
    pose17_to_pose23,
    reconstruct_anchored_pose23,
    to_anchored_relative_pose23,
)
from object_tracking.vla_target_alignment import (
    FUTURE_STATE_TARGET_V1,
    align_achieved_future_state,
    alignment_metadata,
)


def resolve_episode(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (root.parents[1], root.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return root.parents[1] / candidate


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    if values.ndim != 2 or weights.shape != (len(values),):
        raise ValueError("values must be [N,D] with matching [N] weights")
    if not 0.0 <= quantile <= 1.0 or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("invalid weighted quantile inputs")
    output = np.empty(values.shape[1], dtype=np.float64)
    for dimension in range(values.shape[1]):
        order = np.argsort(values[:, dimension], kind="stable")
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        threshold = quantile * cumulative[-1]
        index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1)
        output[dimension] = values[order[index], dimension]
    return output


def summarize(values: np.ndarray, weights: np.ndarray) -> dict[str, list[float]]:
    normalized = weights / np.sum(weights)
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum(np.square(values - mean) * normalized[:, None], axis=0)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
        "q01": weighted_quantile(values, weights, 0.01).tolist(),
        "q99": weighted_quantile(values, weights, 0.99).tolist(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required to read canonical episodes") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--absolute-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--real-weight", type=float, default=0.75)
    parser.add_argument("--roundtrip-samples", type=int, default=4096)
    parser.add_argument(
        "--future-state-lookahead",
        type=int,
        default=0,
        help=(
            "derive each action target from achieved state[t+N] instead of "
            "the source-specific recorded command"
        ),
    )
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if not 0.0 < args.real_weight < 1.0:
        raise ValueError("--real-weight must be between zero and one")
    if args.future_state_lookahead < 0:
        raise ValueError("--future-state-lookahead cannot be negative")

    manifest_path = args.canonical_root / "CANONICAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    absolute_stats = json.loads(args.absolute_stats.read_text(encoding="utf-8"))
    actions_by_source: dict[str, list[np.ndarray]] = {"sim": [], "real": []}
    anchors_by_source: dict[str, list[np.ndarray]] = {"sim": [], "real": []}
    horizon_by_source: dict[str, list[np.ndarray]] = {"sim": [], "real": []}
    source_counts = {
        "sim": {"trajectories": 0, "transitions": 0, "valid_chunk_targets": 0},
        "real": {"trajectories": 0, "transitions": 0, "valid_chunk_targets": 0},
    }

    for entry in manifest["episodes"]:
        if str(entry["split"]).replace("validation", "val") != "train":
            continue
        source = "sim" if entry["source"] == "isaac" else "real"
        path = resolve_episode(args.canonical_root, entry["path"])
        with h5py.File(path, "r") as episode:
            states = pose17_to_pose23(np.asarray(episode["observations/ee_qpos"][:]))
            if args.future_state_lookahead:
                aligned = align_achieved_future_state(
                    states, args.future_state_lookahead
                )
                states = aligned.observations
                targets = aligned.targets
            else:
                targets = pose17_to_pose23(np.asarray(episode["ee_action"][:]))
        if states.shape != targets.shape or len(states) == 0:
            raise ValueError(f"{path}: inconsistent or empty state/action arrays")

        timestep, horizon = np.nonzero(
            np.arange(args.horizon)[None, :] + np.arange(len(states))[:, None]
            < len(states)
        )
        target_indices = timestep + horizon
        anchors = states[timestep]
        absolute_targets = targets[target_indices]
        relative = to_anchored_relative_pose23(anchors, absolute_targets)
        actions_by_source[source].append(relative)
        anchors_by_source[source].append(anchors)
        horizon_by_source[source].append(horizon.astype(np.int16))
        source_counts[source]["trajectories"] += 1
        source_counts[source]["transitions"] += len(states)
        source_counts[source]["valid_chunk_targets"] += len(relative)

    combined: list[np.ndarray] = []
    combined_weights: list[np.ndarray] = []
    per_horizon: dict[str, dict[str, Any]] = {}
    for source, weight in (("sim", 1.0 - args.real_weight), ("real", args.real_weight)):
        if not actions_by_source[source]:
            raise ValueError(f"no train episodes found for {source}")
        source_values = np.concatenate(actions_by_source[source])
        combined.append(source_values)
        combined_weights.append(
            np.full(len(source_values), weight / len(source_values), dtype=np.float64)
        )

    values = np.concatenate(combined).astype(np.float64)
    weights = np.concatenate(combined_weights)
    all_horizons = np.concatenate(
        [np.concatenate(horizon_by_source[source]) for source in ("sim", "real")]
    )
    for horizon in range(args.horizon):
        mask = all_horizons == horizon
        horizon_weights = weights[mask]
        horizon_values = values[mask]
        per_horizon[str(horizon)] = {
            "count": int(np.count_nonzero(mask)),
            "action": summarize(horizon_values, horizon_weights),
            "right_translation_norm": summarize(
                np.linalg.norm(horizon_values[:, 9:12], axis=1)[:, None],
                horizon_weights,
            ),
        }

    rng = np.random.default_rng(20260723)
    sample_count = min(args.roundtrip_samples, len(values))
    indices = rng.choice(len(values), size=sample_count, replace=False)
    all_anchors = np.concatenate(
        [np.concatenate(anchors_by_source[source]) for source in ("sim", "real")]
    )
    reconstructed = reconstruct_anchored_pose23(all_anchors[indices], values[indices])
    # Re-encoding is the stable comparison because rotation-6D inputs are
    # orthonormalized by contract.
    reencoded = to_anchored_relative_pose23(all_anchors[indices], reconstructed)
    roundtrip_max = float(np.max(np.abs(reencoded - values[indices])))
    if roundtrip_max > 1e-5:
        raise ValueError(f"relative round-trip error {roundtrip_max} exceeds tolerance")

    payload = {
        "action": summarize(values, weights),
        "proprio": absolute_stats["proprio"],
        "num_trajectories": sum(item["trajectories"] for item in source_counts.values()),
        "num_transitions": sum(item["transitions"] for item in source_counts.values()),
        "representation": {
            "version": RELATIVE_POSE23_V1,
            "translation_frame": "anchor_torso",
            "rotation_6d": "concat_first_then_second_matrix_columns",
            "rotation_delta": "R_anchor_transpose_times_R_target",
            "anchor": "newest_raw_observation_proprio",
            "horizon": args.horizon,
            "each_horizon_uses_same_anchor": True,
            "padded_targets_in_statistics": False,
            "controlled_dimensions": list(range(9, 18)),
            "held_dimensions": [*range(0, 9), *range(18, 23)],
            "target_alignment": (
                alignment_metadata(args.future_state_lookahead)
                if args.future_state_lookahead
                else {
                    "version": "recorded_command_v1",
                    "source_independent": False,
                }
            ),
        },
        "provenance": {
            "train_only": True,
            "canonical_manifest_sha256": sha256(manifest_path),
            "absolute_statistics_sha256": sha256(args.absolute_stats),
            "source_weights": {"sim": 1.0 - args.real_weight, "real": args.real_weight},
            "source_counts": source_counts,
            "weighting": "equal source mass then equal valid chunk-target mass within source",
            "roundtrip_samples": sample_count,
            "roundtrip_max_abs_error": roundtrip_max,
            "per_horizon": per_horizon,
            "future_state_target_version": (
                FUTURE_STATE_TARGET_V1
                if args.future_state_lookahead
                else None
            ),
        },
    }
    atomic_json(args.output, payload)
    print(
        "RELATIVE_POSE23_STATS_READY "
        f"output={args.output} valid_targets={len(values)} "
        f"roundtrip_max={roundtrip_max:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
