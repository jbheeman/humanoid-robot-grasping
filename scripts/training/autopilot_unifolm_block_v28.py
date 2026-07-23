#!/usr/bin/env python3
"""Train, validate, select, and materialize the block-v28 UniFoLM policy.

The stages run sequentially in one process.  There is no timer polling: each
stage begins only after the preceding subprocess exits successfully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home/aarav/Documents/g1-bunny-vla-workspace")
PYTHON = Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python")
RUNS = ROOT / "runs/unifolm_plush_touch"
LOGS = ROOT / "logs/block_v28"
STATUS = RUNS / "BLOCK_V28_AUTOPILOT_STATUS.json"
SUMMARY = RUNS / "BLOCK_V28_TRAINING_SUMMARY.json"
CONFIG = ROOT / "configs/vla/plush_touch_block_v28_train.yaml"
DATA = ROOT / "datasets/plush_touch_rlds_block_v28"
STATS = ROOT / "datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
BASE = ROOT / "models/pretrained/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt"
LAUNCHER = ROOT / "scripts/run_unifolm_block_v28.sh"
EVALUATOR = ROOT / "scripts/evaluate_plush_touch_checkpoints.py"
MATERIALIZER = ROOT / "scripts/materialize_plush_touch_vla.py"
MIN_FREE_GB = 100.0

STAGES = (
    {
        "mode": "sim",
        "run_id": "block-v28-sim-3k",
        "estimated_output_gb": 9.0,
    },
    {
        "mode": "mixed",
        "run_id": "block-v28-mixed-4k",
        "estimated_output_gb": 12.0,
    },
    {
        "mode": "real",
        "run_id": "block-v28-real-1k",
        "estimated_output_gb": 7.0,
    },
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def update_status(phase: str, **details: object) -> None:
    payload = {
        "phase": phase,
        "updated_unix": time.time(),
        "minimum_free_disk_gb": MIN_FREE_GB,
        **details,
    }
    write_json(STATUS, payload)
    print(f"BLOCK_V28 phase={phase} {json.dumps(details, sort_keys=True)}", flush=True)


def free_disk_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1024**3


def require_disk_budget(new_output_gb: float, phase: str) -> None:
    free_gb = free_disk_gb()
    required_gb = MIN_FREE_GB + new_output_gb
    if free_gb < required_gb:
        raise RuntimeError(
            f"disk reserve blocked {phase}: {free_gb:.1f} GB free; "
            f"{required_gb:.1f} GB required"
        )


def run_logged(command: list[str], log_path: Path, phase: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_status(
        phase,
        command=command,
        log=str(log_path),
        free_disk_gb=round(free_disk_gb(), 1),
    )
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{phase} exited {result.returncode}; inspect {log_path}"
        )


def evaluate(run_dir: Path, label: str, split: str = "val") -> dict:
    report = RUNS / f"block_v28_{label}_{split}.json"
    samples = "64" if split == "test" else "48"
    run_logged(
        [
            str(PYTHON),
            str(EVALUATOR),
            "--run-dir",
            str(run_dir),
            "--output",
            str(report),
            "--samples-per-source",
            samples,
            "--config",
            str(CONFIG),
            "--data-root",
            str(DATA),
            "--stats",
            str(STATS),
            "--split",
            split,
        ],
        LOGS / f"evaluate_{label}_{split}.log",
        f"evaluating_{label}_{split}",
    )
    return json.loads(report.read_text())


def train_stage(stage: dict, action_checkpoint: Path | None) -> Path:
    run_dir = RUNS / stage["run_id"]
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {run_dir}")
    require_disk_budget(stage["estimated_output_gb"], f"train_{stage['mode']}")
    command = [str(LAUNCHER), stage["mode"]]
    if action_checkpoint is not None:
        command.append(str(action_checkpoint))
    run_logged(
        command,
        LOGS / f"train_{stage['mode']}.log",
        f"training_{stage['mode']}",
    )
    final = run_dir / "final_model/action_model.pt"
    if not final.is_file():
        raise RuntimeError(f"stage did not create final checkpoint: {final}")
    return run_dir


def candidate_from_report(report: dict, stage: str) -> dict:
    candidate = dict(report["best"])
    candidate["stage"] = stage
    return candidate


def main() -> int:
    history: list[dict] = []
    selected_for_next: Path | None = None
    try:
        for stage in STAGES:
            run_dir = train_stage(stage, selected_for_next)
            report = evaluate(run_dir, stage["mode"])
            candidate = candidate_from_report(report, stage["mode"])
            history.append(
                {
                    "stage": stage["mode"],
                    "run_dir": str(run_dir),
                    "validation_report": str(
                        RUNS / f"block_v28_{stage['mode']}_val.json"
                    ),
                    "best": candidate,
                }
            )
            selected_for_next = Path(candidate["checkpoint"])

        # Selection uses validation only.  Test remains sealed until this point.
        candidates = [entry["best"] for entry in history]
        selected = min(candidates, key=lambda item: item["score"])
        selected_path = Path(selected["checkpoint"])
        selected_run = selected_path.parents[1]

        test_report = RUNS / "block_v28_selected_test.json"
        run_logged(
            [
                str(PYTHON),
                str(EVALUATOR),
                "--run-dir",
                str(selected_run),
                "--checkpoint",
                str(selected_path),
                "--output",
                str(test_report),
                "--samples-per-source",
                "64",
                "--config",
                str(CONFIG),
                "--data-root",
                str(DATA),
                "--stats",
                str(STATS),
                "--split",
                "test",
            ],
            LOGS / "evaluate_selected_test.log",
            "evaluating_selected_test",
        )

        real_val = selected["sources"]["g1_plush_touch_real"]
        promotion_passed = (
            real_val["right_xyz_ade_m"] < 0.08
            and real_val["right_xyz_fde_m"] < 0.10
        )

        require_disk_budget(22.0, "materialize_selected")
        output_dir = RUNS / "block-v28-selected-merged"
        if output_dir.exists():
            raise RuntimeError(f"refusing to overwrite merged model: {output_dir}")
        run_logged(
            [
                str(PYTHON),
                str(MATERIALIZER),
                "--base",
                str(BASE),
                "--action",
                str(selected_path),
                "--output-dir",
                str(output_dir),
            ],
            LOGS / "materialize_selected.log",
            "materializing_selected",
        )

        summary = {
            "status": "complete",
            "selected": selected,
            "history": history,
            "test_report": str(test_report),
            "merged_vla": str(output_dir / "pytorch_model.pt"),
            "promotion": {
                "passed_offline_gate": promotion_passed,
                "real_validation_right_xyz_ade_limit_m": 0.08,
                "real_validation_right_xyz_fde_limit_m": 0.10,
                "robot_execution_authorized": False,
            },
            "minimum_free_disk_gb": MIN_FREE_GB,
            "final_free_disk_gb": round(free_disk_gb(), 1),
            "robot_contacted": False,
        }
        write_json(SUMMARY, summary)
        update_status(
            "complete",
            summary=str(SUMMARY),
            selected_checkpoint=str(selected_path),
            merged_vla=summary["merged_vla"],
            promotion_passed=promotion_passed,
            free_disk_gb=summary["final_free_disk_gb"],
        )
        return 0
    except Exception as error:
        update_status(
            "failed",
            error=repr(error),
            free_disk_gb=round(free_disk_gb(), 1),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
