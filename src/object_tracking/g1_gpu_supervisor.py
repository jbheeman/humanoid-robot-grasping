"""Adaptive VRAM supervisor for bounded MJX G1 simulation batches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from object_tracking.g1_sim_cli import root


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
    output = root() / "runs/simulation-gpu/adaptive-status.json"
    log_dir = root() / "runs/simulation-gpu/adaptive"
    log_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.initial_candidates
    batch = 0
    started = time.time()
    stop_at = started + args.hours * 3600
    history: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    while time.time() < stop_at:
        before, total = read_vram(args.nvidia_smi)
        if total < args.target_total_mib:
            raise RuntimeError("target VRAM exceeds the detected GPU capacity")
        if before >= args.target_total_mib - args.launch_headroom_mib:
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
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            peak = before
            aborted = False
            while process.poll() is None:
                time.sleep(args.poll_seconds)
                used, _ = read_vram(args.nvidia_smi)
                peak = max(peak, used)
                if used >= args.target_total_mib:
                    aborted = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            return_code = process.wait()
        after, _ = read_vram(args.nvidia_smi)
        entry = {
            "batch": batch,
            "candidates": candidates,
            "steps": args.steps,
            "vram_before_mib": before,
            "vram_peak_mib": peak,
            "vram_after_mib": after,
            "aborted_for_vram": aborted,
            "return_code": return_code,
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
            "detected_total_vram_mib": total,
            "next_candidates": candidates,
            "elapsed_hours": (time.time() - started) / 3600,
            "history": history,
        }
        _write(output, status)
        print(
            f"batch={batch} candidates={entry['candidates']} peak={peak}MiB "
            f"after={after}MiB rc={return_code} next={candidates}",
            flush=True,
        )
    status["status"] = "complete"
    _write(output, status)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--initial-candidates", type=int, default=64)
    p.add_argument("--min-candidates", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=1024)
    p.add_argument("--target-total-mib", type=int, default=11264)
    p.add_argument("--launch-headroom-mib", type=int, default=512)
    p.add_argument("--soft-headroom-mib", type=int, default=512)
    p.add_argument("--poll-seconds", type=float, default=0.25)
    p.add_argument("--jax-memory-fraction", type=float, default=0.55)
    p.add_argument("--nvidia-smi", default="/usr/lib/wsl/lib/nvidia-smi")
    p.set_defaults(func=run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
