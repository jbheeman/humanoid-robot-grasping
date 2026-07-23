#!/usr/bin/env python3
"""Evaluate matched future-state lookahead pilots on validation only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from typing import Any


VARIANTS = ("future1-t1-right9", "future3-t1-right9")


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
    parser.add_argument("--gpus", type=int, nargs=2, default=(0, 1))
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain two distinct non-negative indices")

    python = Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python")
    config_root = args.root / "configs/vla/latency_pilots"
    output_dir = args.root / "runs/diagnostics/pose23_latency"
    output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(variant: str, gpu: int) -> dict[str, Any]:
        config = config_root / f"{variant}.yaml"
        config_payload = __import__("yaml").safe_load(
            config.read_text(encoding="utf-8")
        )
        data = config_payload["datasets"]["vla_data"]
        run_id = config_payload["run_id"]
        checkpoint = (
            args.root
            / "runs/unifolm_plush_touch"
            / run_id
            / "checkpoints/steps_750_action_model.pt"
        )
        report = output_dir / f"{variant}_val.json"
        log = output_dir / f"{variant}_val.log"
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
            str(data["data_root_dir"]),
            "--stats",
            str(data["relative_action_statistics"]),
            "--split",
            "val",
            "--skip-visual-perturbations",
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
            "gpu": gpu,
            "score": weighted_ade(payload),
            "gate_passed": bool(payload["gate"]["passed"]),
            "report": str(report),
            "sources": payload["sources"],
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(evaluate, variant, gpu)
            for variant, gpu in zip(VARIANTS, args.gpus, strict=True)
        ]
        records = [future.result() for future in futures]
    records.sort(key=lambda item: VARIANTS.index(item["variant"]))
    selected = min(records, key=lambda item: item["score"])
    prior_report = json.loads(
        (
            args.root
            / "runs/diagnostics/pose23_pilots/relative-t1-right9_val.json"
        ).read_text(encoding="utf-8")
    )
    prior_score = weighted_ade(prior_report)
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "records": records,
        "prior_750_step_relative_score": prior_score,
        "selected": selected,
        "promotion": {
            "beats_prior_relative_pilot": selected["score"] < prior_score,
            "selected_passes_baseline_gate": bool(selected["gate_passed"]),
            "requires_closed_loop_simulation": True,
            "robot_execution_authorized": False,
        },
        "physical_robot_contacted": False,
    }
    output = output_dir / "SELECTION.json"
    atomic_json(output, payload)
    print(
        "POSE23_LATENCY_SELECTION "
        f"selected={selected['variant']} score={selected['score']:.6f} "
        f"prior={prior_score:.6f} gate={int(selected['gate_passed'])} "
        f"output={output}"
    )
    return (
        0
        if selected["score"] < prior_score and selected["gate_passed"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
