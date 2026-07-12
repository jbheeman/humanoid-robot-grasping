"""Adaptive VRAM supervisor for bounded MJX G1 simulation batches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from object_tracking.g1_sim_cli import root

_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _terminate_active(_signum: int, _frame: Any) -> None:
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
    raise SystemExit(128 + _signum)


def read_vram(command: str) -> tuple[int, int]:
    completed = subprocess.run(
        [command, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()[0].split(",")
    return int(values[0].strip()), int(values[1].strip())


def next_candidate_count(
    current: int,
    *,
    peak_mib: int,
    target_mib: int,
    aborted: bool,
    minimum: int,
    maximum: int,
) -> int:
    """Scale conservatively; a killed batch always backs off substantially."""
    if aborted:
        return max(minimum, int(current * 0.60))
    spare = target_mib - peak_mib
    if spare < 512:
        return current
    factor = 1.05 if spare < 1024 else (1.15 if spare < 2048 else 1.35)
    return min(maximum, max(current + 1, int(current * factor)))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    global _ACTIVE_PROCESS
    signal.signal(signal.SIGTERM, _terminate_active)
    signal.signal(signal.SIGINT, _terminate_active)
    output = root() / "runs/simulation-gpu/adaptive-status.json"
    best_path = root() / "runs/simulation-gpu/best-checkpoint.json"
    latest_path = root() / "runs/simulation-gpu/latest.json"
    log_dir = root() / "runs/simulation-gpu/adaptive"
    log_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.initial_candidates
    batch = 0
    started = time.time()
    stop_at = started + args.hours * 3600
    history: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")
    if best_path.is_file():
        best_payload = json.loads(best_path.read_text())
        best_score = float(best_payload["best_normalized_p95_rmse"])
    last_improvement = started
    stop_reason = "maximum_hours"
    while time.time() < stop_at:
        if time.time() - last_improvement >= args.plateau_hours * 3600:
            stop_reason = "no_improvement"
            break
        before, total = read_vram(args.nvidia_smi)
        pressure_limit = args.target_total_mib - args.soft_headroom_mib
        if total < args.target_total_mib:
            raise RuntimeError("target VRAM exceeds the detected GPU capacity")
        if before >= pressure_limit - args.launch_headroom_mib:
            time.sleep(args.poll_seconds)
            continue
        batch += 1
        log_path = log_dir / f"batch-{batch:05d}.log"
        child_env = os.environ.copy()
        child_env["G1_GPU_MEMORY_FRACTION"] = str(args.jax_memory_fraction)
        with log_path.open("w") as stream:
            process = subprocess.Popen(
                [
                    str(root() / "scripts/sim/gpu-run.sh"),
                    "benchmark",
                    "--candidates",
                    str(candidates),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(args.seed + batch),
                    "--checkpoint",
                    str(best_path),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
                start_new_session=True,
            )
            _ACTIVE_PROCESS = process
            peak = before
            aborted = False
            while process.poll() is None:
                time.sleep(args.poll_seconds)
                used, _ = read_vram(args.nvidia_smi)
                peak = max(peak, used)
                if used >= pressure_limit:
                    aborted = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
            return_code = process.wait()
            _ACTIVE_PROCESS = None
        after, _ = read_vram(args.nvidia_smi)
        improved = False
        meaningful_improvement = False
        if not aborted and return_code == 0 and latest_path.is_file():
            result = json.loads(latest_path.read_text())
            score = float(result["best_normalized_p95_rmse"])
            if score < best_score:
                meaningful_improvement = score < best_score * (
                    1.0 - args.minimum_relative_improvement
                )
                best_score = score
                best_payload = result
                best_payload["checkpoint_batch"] = batch
                best_payload["checkpointed_at"] = datetime.now(timezone.utc).isoformat()
                _write(best_path, best_payload)
                improved = True
                if meaningful_improvement:
                    last_improvement = time.time()
        entry = {
            "batch": batch,
            "candidates": candidates,
            "steps": args.steps,
            "vram_before_mib": before,
            "vram_peak_mib": peak,
            "vram_after_mib": after,
            "aborted_for_vram": aborted,
            "return_code": return_code,
            "improved_global_best": improved,
            "meaningful_improvement": meaningful_improvement,
            "global_best_score": None if best_payload is None else best_score,
            "log": str(log_path.relative_to(root())),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        history = history[-100:]
        candidates = next_candidate_count(
            candidates,
            peak_mib=peak,
            target_mib=args.target_total_mib - args.soft_headroom_mib,
            aborted=aborted or return_code != 0,
            minimum=args.min_candidates,
            maximum=args.max_candidates,
        )
        status = {
            "schema_version": 1,
            "status": "running",
            "target_total_vram_mib": args.target_total_mib,
            "pressure_limit_vram_mib": pressure_limit,
            "detected_total_vram_mib": total,
            "next_candidates": candidates,
            "elapsed_hours": (time.time() - started) / 3600,
            "hours_since_improvement": (time.time() - last_improvement) / 3600,
            "best_checkpoint": str(best_path.relative_to(root())),
            "global_best_score": None if best_payload is None else best_score,
            "history": history,
        }
        _write(output, status)
        print(
            f"batch={batch} candidates={entry['candidates']} peak={peak}MiB "
            f"after={after}MiB rc={return_code} next={candidates} "
            f"best={best_score:.6f} improved={improved}",
            flush=True,
        )
    status["status"] = "complete"
    status["stop_reason"] = stop_reason
    _write(output, status)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--plateau-hours", type=float, default=1.0)
    p.add_argument("--minimum-relative-improvement", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--initial-candidates", type=int, default=64)
    p.add_argument("--min-candidates", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=1024)
    p.add_argument("--target-total-mib", type=int, default=12000)
    p.add_argument("--launch-headroom-mib", type=int, default=256)
    p.add_argument("--soft-headroom-mib", type=int, default=512)
    p.add_argument("--poll-seconds", type=float, default=0.25)
    p.add_argument("--jax-memory-fraction", type=float, default=0.90)
    p.add_argument("--nvidia-smi", default="/usr/lib/wsl/lib/nvidia-smi")
    p.set_defaults(func=run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
