#!/usr/bin/env python3
"""Relay compact simulation elites between isolated CPU and GPU workstations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXCHANGE = ROOT / "simulation/exchange/elite_candidates.json"
CPU_HOST = "software@100.64.0.25"
CPU_ROOT = "~/Documents/utils/aarav/humanoid-robot-grasping"
GPU_HOST = "theaa@10.0.0.65"
GPU_ROOT = "~/Documents/coding/humanoid-robot-grasping"


def _remote_json(host: str, command: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["ssh", host, command], check=True, capture_output=True, text=True, timeout=20
        )
        return json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def collect() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    cpu = _remote_json(CPU_HOST, f"cd {CPU_ROOT} && scripts/sim/run.sh status")
    if cpu and isinstance(cpu.get("best_candidate"), dict):
        candidates.append(
            {
                **cpu["best_candidate"],
                "source": "cpu_mujoco",
                "source_score": cpu.get("best_score"),
            }
        )
    gpu = _remote_json(GPU_HOST, f"cd {GPU_ROOT} && cat runs/simulation-gpu/best-checkpoint.json")
    if gpu:
        elites = gpu.get("elite_candidates") or [gpu.get("best_candidate", {})]
        candidates.extend(
            {
                **candidate,
                "source": "gpu_mjx",
                "source_score": gpu.get("best_normalized_p95_rmse"),
            }
            for candidate in elites
            if isinstance(candidate, dict)
        )
    return candidates[:16]


def publish(candidates: list[dict[str, Any]]) -> bool:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
    }
    previous = json.loads(EXCHANGE.read_text()) if EXCHANGE.exists() else {}
    if previous.get("candidates") == candidates:
        return False
    EXCHANGE.parent.mkdir(parents=True, exist_ok=True)
    temporary = EXCHANGE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(EXCHANGE)
    subprocess.run(["git", "add", str(EXCHANGE)], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Exchange simulation elites"], cwd=ROOT, check=True
    )
    subprocess.run(["git", "push", "origin", "sim"], cwd=ROOT, check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        changed = publish(collect())
        print(f"{datetime.now().isoformat()} exchange_updated={changed}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
