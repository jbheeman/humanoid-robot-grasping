#!/usr/bin/env python3
"""Evaluate matched absolute/relative Pose23 pilots on validation only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from typing import Any


VARIANTS = (
    "absolute-t1-right9",
    "relative-t1-right9",
    "absolute-t5s3-right9",
    "relative-t5s3-right9",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def weighted_ade(report: dict[str, Any]) -> float:
    sources = report["sources"]
    return float(
        0.75 * sources["g1_plush_touch_real"]["model"]["ade_m"]
        + 0.25 * sources["g1_plush_touch_sim"]["model"]["ade_m"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/aarav/Documents/g1-bunny-vla-workspace"),
    )
    parser.add_argument("--samples-per-source", type=int, default=96)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("ABSOLUTE_GPU", "RELATIVE_GPU"),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse completed per-variant reports and evaluate only missing variants",
    )
    args = parser.parse_args()

    python = Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python")
    data_root = args.root / "datasets/plush_touch_rlds_block_v28"
    stats_root = args.root / "datasets/plush_touch_canonical_block_v28"
    absolute_stats = stats_root / "SHARED_TRAIN_STATS_75_REAL.json"
    relative_stats = (
        stats_root / "SHARED_TRAIN_STATS_75_REAL_RELATIVE_POSE23_V1.json"
    )
    output_dir = args.root / "runs/diagnostics/pose23_pilots"
    output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(variant: str, gpu: int) -> dict[str, Any]:
        representation = variant.split("-", 1)[0]
        stats = relative_stats if representation == "relative" else absolute_stats
        checkpoint = (
            args.root
            / "runs/unifolm_plush_touch"
            / f"pose23-pilot-{variant}"
            / "checkpoints/steps_750_action_model.pt"
        )
        config = args.root / "configs/vla/relative_pilots" / f"{variant}.yaml"
        report = output_dir / f"{variant}_val.json"
        log = output_dir / f"{variant}_val.log"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")
        if args.reuse_existing and report.is_file():
            payload = json.loads(report.read_text(encoding="utf-8"))
            if payload.get("split") != "val":
                raise RuntimeError(f"existing report is not validation-only: {report}")
            return {
                "variant": variant,
                "representation": representation,
                "gpu": None,
                "score": weighted_ade(payload),
                "gate_passed": bool(payload["gate"]["passed"]),
                "report": str(report),
                "sources": payload["sources"],
                "reused": True,
            }
        command = [
            str(python),
            str(args.root / "scripts/diagnose_unifolm_predictions.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(report),
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
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(args.root / "src"),
                str(args.root / "unifolm-vla/src"),
                environment.get("PYTHONPATH", ""),
            )
        )
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=args.root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode not in (0, 2) or not report.is_file():
            raise RuntimeError(
                f"{variant} diagnostic exited {result.returncode}; inspect {log}"
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        return {
            "variant": variant,
            "representation": representation,
            "gpu": gpu,
            "score": weighted_ade(payload),
            "gate_passed": bool(payload["gate"]["passed"]),
            "report": str(report),
            "sources": payload["sources"],
            "reused": False,
        }

    if len(set(args.gpus)) != 2 or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain two distinct non-negative indices")
    assignments = {
        args.gpus[0]: ("absolute-t1-right9", "absolute-t5s3-right9"),
        args.gpus[1]: ("relative-t1-right9", "relative-t5s3-right9"),
    }
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                lambda gpu=gpu, variants=variants: [
                    evaluate(variant, gpu) for variant in variants
                ]
            )
            for gpu, variants in assignments.items()
        ]
        for future in futures:
            records.extend(future.result())
    records.sort(key=lambda item: VARIANTS.index(item["variant"]))

    by_name = {item["variant"]: item for item in records}
    matched = {}
    for temporal in ("t1", "t5s3"):
        absolute = by_name[f"absolute-{temporal}-right9"]["score"]
        relative = by_name[f"relative-{temporal}-right9"]["score"]
        matched[temporal] = {
            "absolute_score": absolute,
            "relative_score": relative,
            "relative_improvement_fraction": (absolute - relative) / absolute,
        }
    selected = min(records, key=lambda item: item["score"])
    relative_wins_both = all(
        comparison["relative_improvement_fraction"] > 0
        for comparison in matched.values()
    )
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "samples_per_source": args.samples_per_source,
        "records": records,
        "matched": matched,
        "selected": selected,
        "promotion": {
            "relative_wins_both_matched_pilots": relative_wins_both,
            "selected_passes_baseline_gate": bool(selected["gate_passed"]),
            "requires_closed_loop_simulation": True,
            "robot_execution_authorized": False,
        },
        "physical_robot_contacted": False,
    }
    output = output_dir / "SELECTION.json"
    atomic_json(output, payload)
    print(
        "POSE23_PILOT_SELECTION "
        f"selected={selected['variant']} score={selected['score']:.6f} "
        f"relative_wins_both={int(relative_wins_both)} output={output}"
    )
    return 0 if relative_wins_both and selected["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
