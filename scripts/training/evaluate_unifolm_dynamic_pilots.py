#!/usr/bin/env python3
"""Evaluate temporal/action-loss pilots and select on validation only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from typing import Any


VARIANTS = ("t1-all23", "t1-right9", "t5s3-right9")
VARIANT_COST = {"t1-all23": 1, "t1-right9": 1, "t5s3-right9": 2}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def score(report: dict[str, Any]) -> float:
    sources = report["sources"]
    return float(
        0.75 * sources["g1_plush_touch_real"]["model"]["ade_m"]
        + 0.25 * sources["g1_plush_touch_sim"]["model"]["ade_m"]
    )


def assign_variants_to_gpus(
    variants: tuple[str, ...], gpus: tuple[int, ...]
) -> dict[int, tuple[str, ...]]:
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be unique non-negative integers")
    queues: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    loads = {gpu: 0 for gpu in gpus}
    for variant in sorted(variants, key=lambda name: VARIANT_COST[name], reverse=True):
        gpu = min(gpus, key=lambda index: (loads[index], index))
        queues[gpu].append(variant)
        loads[gpu] += VARIANT_COST[variant]
    return {gpu: tuple(queue) for gpu, queue in queues.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/aarav/Documents/g1-bunny-vla-workspace"),
    )
    parser.add_argument("--samples-per-source", type=int, default=48)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    args = parser.parse_args()
    gpus = tuple(args.gpus)
    queues = assign_variants_to_gpus(VARIANTS, gpus)

    python = Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python")
    data_root = args.root / "datasets/plush_touch_rlds_block_v28"
    stats = (
        args.root
        / "datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
    )
    reports_dir = args.root / "runs/diagnostics/dynamic_pilots"
    reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_variant(variant: str, gpu: int) -> dict[str, Any]:
        checkpoint = (
            args.root
            / f"runs/unifolm_plush_touch/dynamic-pilot-{variant}"
            / "checkpoints/steps_750_action_model.pt"
        )
        config = args.root / f"configs/vla/dynamic_pilots/{variant}.yaml"
        report_path = reports_dir / f"{variant}_val.json"
        log_path = reports_dir / f"{variant}_val.log"
        if not checkpoint.is_file():
            raise RuntimeError(f"pilot checkpoint is missing: {checkpoint}")
        command = [
            str(python),
            str(args.root / "scripts/diagnose_unifolm_predictions.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(report_path),
            "--samples-per-source",
            str(args.samples_per_source),
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--stats",
            str(stats),
            "--split",
            "val",
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=args.root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        # Exit 2 is the evaluator's expected "did not beat baseline" gate.
        if result.returncode not in (0, 2) or not report_path.is_file():
            raise RuntimeError(
                f"diagnostic failed for {variant} with exit {result.returncode}; "
                f"inspect {log_path}"
            )
        report = json.loads(report_path.read_text())
        return {
            "variant": variant,
            "gpu": gpu,
            "checkpoint": str(checkpoint),
            "config": str(config),
            "report": str(report_path),
            "score": score(report),
            "gate_passed": bool(report["gate"]["passed"]),
            "window_size": report["window_size"],
            "observation_stride": report["observation_stride"],
            "history_span_s_at_30hz": report["history_span_s_at_30hz"],
            "sources": report["sources"],
        }

    def evaluate_queue(gpu: int, variants: tuple[str, ...]) -> list[dict[str, Any]]:
        return [evaluate_variant(variant, gpu) for variant in variants]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [
            executor.submit(evaluate_queue, gpu, variants)
            for gpu, variants in queues.items()
            if variants
        ]
        for future in futures:
            records.extend(future.result())
    records.sort(key=lambda record: VARIANTS.index(record["variant"]))

    selected = min(records, key=lambda record: record["score"])
    gate_passed = bool(selected["gate_passed"])
    summary = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "gpu_queues": {
            str(gpu): list(variants) for gpu, variants in queues.items()
        },
        "score": "0.75*real_right_xyz_ADE + 0.25*sim_right_xyz_ADE",
        "records": records,
        "selected": selected,
        "promotion": {
            "passed_offline_gate": gate_passed,
            "requires_non_paused_simulation": True,
            "robot_execution_authorized": False,
        },
        "physical_robot_contacted": False,
    }
    output = reports_dir / "SELECTION.json"
    write_json(output, summary)
    print(
        "DYNAMIC_PILOT_SELECTION",
        f"selected={selected['variant']}",
        f"score={selected['score']:.6f}",
        f"gate_passed={int(gate_passed)}",
        f"output={output}",
    )
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
