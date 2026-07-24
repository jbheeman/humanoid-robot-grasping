#!/usr/bin/env python3
"""Select and gate a completed UniFoLM run without leaking the test split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"steps_(\d+)_action_model\.pt$")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a UniFoLM action checkpoint: {path}")
    return int(match.group(1))


def discover_checkpoints(run_dir: Path, expected_final_step: int) -> list[Path]:
    checkpoints = sorted(
        run_dir.glob("checkpoints/steps_*_action_model.pt"),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise RuntimeError(f"no action checkpoints found under {run_dir}")
    steps = [checkpoint_step(path) for path in checkpoints]
    if len(steps) != len(set(steps)):
        raise RuntimeError(f"duplicate checkpoint steps: {steps}")
    if steps[-1] != expected_final_step:
        raise RuntimeError(
            f"training incomplete: final checkpoint is {steps[-1]}, "
            f"expected {expected_final_step}"
        )
    empty = [str(path) for path in checkpoints if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"empty checkpoints: {empty}")
    return checkpoints


def weighted_ade(report: dict[str, Any], real_weight: float) -> float:
    sources = report["sources"]
    real = float(sources["g1_plush_touch_real"]["model"]["ade_m"])
    sim = float(sources["g1_plush_touch_sim"]["model"]["ade_m"])
    return real_weight * real + (1.0 - real_weight) * sim


def validation_gate(report: dict[str, Any]) -> dict[str, Any]:
    sources = report["sources"]
    visual = {
        source: bool(metrics["diagnosis_flags"]["visually_conditioned"])
        for source, metrics in sources.items()
    }
    saturation_ok = {
        source: not bool(
            metrics["diagnosis_flags"]["normalized_output_saturation_over_5_percent"]
        )
        for source, metrics in sources.items()
    }
    magnitude_ok = {
        source: not bool(
            metrics["diagnosis_flags"]["action_magnitude_over_2x_target"]
        )
        for source, metrics in sources.items()
    }
    baseline = bool(report["gate"]["passed"])
    passed = (
        baseline
        and all(visual.values())
        and all(saturation_ok.values())
        and all(magnitude_ok.values())
    )
    return {
        "passed": passed,
        "beats_baselines": baseline,
        "visually_conditioned": visual,
        "saturation_ok": saturation_ok,
        "action_magnitude_ok": magnitude_ok,
    }


class Evaluator:
    def __init__(
        self,
        *,
        root: Path,
        config: Path,
        data_root: Path,
        statistics: Path,
        output_dir: Path,
        samples_per_source: int,
        gate_scope: str,
    ) -> None:
        self.root = root
        self.config = config
        self.data_root = data_root
        self.statistics = statistics
        self.output_dir = output_dir
        self.samples_per_source = samples_per_source
        self.gate_scope = gate_scope
        self.python = Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python")

    def run(
        self,
        checkpoint: Path,
        *,
        split: str,
        gpu: int,
        label: str,
        fast: bool,
    ) -> dict[str, Any]:
        report = self.output_dir / f"{label}_{split}.json"
        log = self.output_dir / f"{label}_{split}.log"
        command = [
            str(self.python),
            str(self.root / "scripts/diagnose_unifolm_predictions.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(report),
            "--samples-per-source",
            str(self.samples_per_source),
            "--config",
            str(self.config),
            "--data-root",
            str(self.data_root),
            "--stats",
            str(self.statistics),
            "--split",
            split,
            "--gate-scope",
            self.gate_scope,
        ]
        if fast:
            command.append("--skip-visual-perturbations")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(self.root / "src"),
                str(self.root / "unifolm-vla/src"),
                environment.get("PYTHONPATH", ""),
            )
        )
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=self.root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode not in (0, 2) or not report.is_file():
            raise RuntimeError(
                f"diagnostic exited {result.returncode}; inspect {log}"
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("split") != split:
            raise RuntimeError(f"{report}: expected split {split}")
        if Path(payload["checkpoint"]).resolve() != checkpoint.resolve():
            raise RuntimeError(f"{report}: checkpoint provenance mismatch")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--samples-per-source", type=int, default=96)
    parser.add_argument("--real-weight", type=float, default=0.75)
    parser.add_argument("--gate-scope", choices=("all", "active"), default="all")
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    args = parser.parse_args()
    if not 0.0 < args.real_weight < 1.0:
        raise ValueError("--real-weight must be between zero and one")
    if len(args.gpus) < 1 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must be unique")

    root = args.root.resolve()
    run_dir = root / "runs/unifolm_plush_touch" / args.run_id
    output_dir = root / "runs/diagnostics" / args.run_id
    status_path = output_dir / "PIPELINE_STATUS.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("complete") is True:
            print(
                "UNIFOLM_PIPELINE_ALREADY_COMPLETE "
                f"phase={previous.get('phase')} status={status_path}"
            )
            return 0 if previous.get("offline_promotion_passed") is True else 2
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "starting_validation",
        "complete": False,
        "test_split_used": False,
        "robot_execution_authorized": False,
        "updated_unix": time.time(),
    }
    atomic_json(status_path, status)
    try:
        checkpoints = discover_checkpoints(run_dir, args.expected_final_step)
        evaluator = Evaluator(
            root=root,
            config=args.config.resolve(),
            data_root=args.data_root.resolve(),
            statistics=args.statistics.resolve(),
            output_dir=output_dir,
            samples_per_source=args.samples_per_source,
            gate_scope=args.gate_scope,
        )

        def evaluate_shard(gpu: int, paths: list[Path]) -> list[dict[str, Any]]:
            records = []
            for checkpoint in paths:
                step = checkpoint_step(checkpoint)
                report = evaluator.run(
                    checkpoint,
                    split="val",
                    gpu=gpu,
                    label=f"step_{step}_fast",
                    fast=True,
                )
                records.append(
                    {
                        "step": step,
                        "checkpoint": str(checkpoint.resolve()),
                        "score": weighted_ade(report, args.real_weight),
                        "report": str(
                            (output_dir / f"step_{step}_fast_val.json").resolve()
                        ),
                    }
                )
            return records

        shards = [checkpoints[index :: len(args.gpus)] for index in range(len(args.gpus))]
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [
                executor.submit(evaluate_shard, gpu, shard)
                for gpu, shard in zip(args.gpus, shards, strict=True)
                if shard
            ]
            records = [record for future in futures for record in future.result()]
        records.sort(key=lambda item: item["step"])
        selected = min(records, key=lambda item: item["score"])
        selected_checkpoint = Path(selected["checkpoint"])

        status.update(
            {
                "phase": "strong_validation",
                "validation_records": records,
                "selected": selected,
                "updated_unix": time.time(),
            }
        )
        atomic_json(status_path, status)
        strong_validation = evaluator.run(
            selected_checkpoint,
            split="val",
            gpu=args.gpus[0],
            label=f"selected_step_{selected['step']}_strong",
            fast=False,
        )
        gate = validation_gate(strong_validation)
        status["validation_gate"] = gate
        status["strong_validation_report"] = str(
            (
                output_dir
                / f"selected_step_{selected['step']}_strong_val.json"
            ).resolve()
        )
        if not gate["passed"]:
            status.update(
                {
                    "phase": "validation_rejected",
                    "complete": True,
                    "test_split_used": False,
                    "updated_unix": time.time(),
                }
            )
            atomic_json(status_path, status)
            print(
                "UNIFOLM_VALIDATION_REJECTED "
                f"step={selected['step']} status={status_path}"
            )
            return 2

        status.update(
            {
                "phase": "sealed_test",
                "test_split_used": True,
                "updated_unix": time.time(),
            }
        )
        atomic_json(status_path, status)
        test_report = evaluator.run(
            selected_checkpoint,
            split="test",
            gpu=args.gpus[0],
            label=f"selected_step_{selected['step']}",
            fast=False,
        )
        test_gate = validation_gate(test_report)
        status.update(
            {
                "phase": "complete" if test_gate["passed"] else "test_rejected",
                "complete": True,
                "test_gate": test_gate,
                "test_report": str(
                    (
                        output_dir
                        / f"selected_step_{selected['step']}_test.json"
                    ).resolve()
                ),
                "offline_promotion_passed": bool(test_gate["passed"]),
                "robot_execution_authorized": False,
                "updated_unix": time.time(),
            }
        )
        atomic_json(status_path, status)
        print(
            "UNIFOLM_PIPELINE_COMPLETE "
            f"step={selected['step']} promotion={int(test_gate['passed'])} "
            f"status={status_path}"
        )
        return 0 if test_gate["passed"] else 2
    except Exception as exc:
        status.update(
            {
                "phase": "failed",
                "complete": True,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_unix": time.time(),
            }
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
