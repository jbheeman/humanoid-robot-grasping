#!/usr/bin/env python3
"""Train a compact 3D bunny trajectory predictor from recorded robot telemetry."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from object_tracking.synthetic_trajectory import synthetic_batch
from object_tracking.trajectory_forecaster import trajectory_features
from object_tracking.trajectory_training import TrajectoryGRU


@dataclass(frozen=True)
class Observation:
    time_s: float
    track_id: int
    position_m: np.ndarray


def parse_horizons(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("horizons must be comma-separated positive seconds")
    return result


def session_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                tracking = row.get("arm_tracking") or {}
                point = np.asarray(tracking.get("object_xyz_m"), dtype=np.float32)
                timestamp = float(row["session_elapsed_s"])
                track_id = int(tracking["track_id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if point.shape == (3,) and np.all(np.isfinite(point)) and timestamp >= 0:
                observations.append(Observation(timestamp, track_id, point))
    return observations


def contiguous_segments(observations: Iterable[Observation], max_gap_s: float) -> list[list[Observation]]:
    result: list[list[Observation]] = []
    current: list[Observation] = []
    for item in observations:
        if current and (item.track_id != current[-1].track_id or item.time_s - current[-1].time_s > max_gap_s):
            result.append(current)
            current = []
        if not current or item.time_s > current[-1].time_s:
            current.append(item)
    if current:
        result.append(current)
    return result


def windows(segment: list[Observation], history: int, horizons: tuple[float, ...], tolerance_s: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if len(segment) < history + 1:
        return samples
    times = np.asarray([item.time_s for item in segment], dtype=np.float32)
    positions = np.asarray([item.position_m for item in segment], dtype=np.float32)
    for end in range(history - 1, len(segment)):
        target_indices = []
        for horizon in horizons:
            expected = times[end] + horizon
            index = int(np.searchsorted(times, expected))
            candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(times)]
            if not candidates:
                break
            best = min(candidates, key=lambda candidate: abs(float(times[candidate] - expected)))
            if abs(float(times[best] - expected)) > tolerance_s:
                break
            target_indices.append(best)
        if len(target_indices) != len(horizons):
            continue
        history_times = times[end - history + 1 : end + 1]
        history_positions = positions[end - history + 1 : end + 1]
        features = trajectory_features(history_times, history_positions)
        targets = positions[target_indices] - positions[end]
        samples.append((features, targets, positions[end]))
    return samples


def collect(root: Path, history: int, horizons: tuple[float, ...]) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    output: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for telemetry in sorted(root.glob("*/telemetry.jsonl")):
        rows = session_observations(telemetry)
        session = []
        for segment in contiguous_segments(rows, max_gap_s=0.20):
            session.extend(windows(segment, history, horizons, tolerance_s=0.050))
        if session:
            output[telemetry.parent.name] = session
    return output


def baseline_mae(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], horizons: tuple[float, ...]) -> float:
    errors = []
    for features, target, _ in samples:
        velocity = features[-1, 3:]
        prediction = np.asarray(horizons, dtype=np.float32)[:, None] * velocity[None, :]
        errors.append(np.abs(prediction - target).mean())
    return float(np.mean(errors)) if errors else float("inf")


def synthetic_baseline_mae(features: np.ndarray, targets: np.ndarray, horizons: tuple[float, ...]) -> float:
    predicted = np.asarray(horizons, dtype=np.float32)[None, :, None] * features[:, -1:, 3:]
    return float(np.abs(predicted - targets).mean())


def train_synthetic(args: argparse.Namespace) -> None:
    """Pretrain from procedural motion without retaining a large dataset."""
    import time

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryGRU(hidden_size=args.hidden_size, outputs=len(args.horizons)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    deadline = time.monotonic() + args.hours * 3600
    best_state = None
    best_mae = float("inf")
    best_baseline = float("inf")
    step = 0
    args.output.mkdir(parents=True, exist_ok=True)
    validation = [
        synthetic_batch(
            validation_rng, batch_size=args.batch, history=args.history, horizons_s=args.horizons
        )
        for _ in range(args.validation_batches)
    ]

    def make_batch(batch_seed: int) -> tuple[np.ndarray, np.ndarray]:
        return synthetic_batch(
            np.random.default_rng(batch_seed),
            batch_size=args.batch,
            history=args.history,
            horizons_s=args.horizons,
        )

    next_seed = args.seed + 10_000
    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as pool:
        pending = [pool.submit(make_batch, next_seed + index) for index in range(args.prefetch_workers * 2)]
        next_seed += len(pending)
        while time.monotonic() < deadline and (args.max_steps <= 0 or step < args.max_steps):
            future = pending.pop(0)
            features, targets = future.result()
            pending.append(pool.submit(make_batch, next_seed))
            next_seed += 1
            x = torch.from_numpy(features).to(device, non_blocking=device.type == "cuda")
            y = torch.from_numpy(targets).to(device, non_blocking=device.type == "cuda")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.eval_every:
                continue

            model.eval()
            errors: list[float] = []
            baselines: list[float] = []
            with torch.inference_mode():
                for val_features, val_targets in validation:
                    prediction = model(torch.from_numpy(val_features).to(device)).cpu().numpy()
                    errors.append(float(np.abs(prediction - val_targets).mean()))
                    baselines.append(synthetic_baseline_mae(val_features, val_targets, args.horizons))
            mae = float(np.mean(errors))
            baseline = float(np.mean(baselines))
            if mae < best_mae:
                best_mae, best_baseline = mae, baseline
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                temporary_checkpoint = args.output / "best.pt.tmp"
                torch.save(
                    {
                        "model_state": best_state,
                        "history_length": args.history,
                        "horizons_s": list(args.horizons),
                        "position_scale_m": 1.0,
                        "max_speed_mps": 2.0,
                        "hidden_size": args.hidden_size,
                    },
                    temporary_checkpoint,
                )
                temporary_checkpoint.replace(args.output / "best.pt")
            print(json.dumps({"step": step, "validation_mae_m": mae, "baseline_mae_m": baseline}), flush=True)

    if best_state is None:
        raise SystemExit("synthetic training ended before its first validation")
    report = {
        "checkpoint": str(args.output / "best.pt"),
        "mode": "synthetic_pretrain",
        "steps": step,
        "validation_mae_m": best_mae,
        "baseline_mae_m": best_baseline,
        "promotable": best_mae < best_baseline,
        "real_data_required": True,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=Path("runs/research/arm_tracking"))
    parser.add_argument("--output", type=Path, default=Path("models/plushie_detector/trajectory_forecaster"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizons", type=parse_horizons, default=(0.10, 0.15, 0.25))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--min-windows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--synthetic-pretrain", action="store_true")
    parser.add_argument("--hours", type=float, default=17.0)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--validation-batches", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=0, help="Test-only cap; zero runs until deadline")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    args = parser.parse_args()
    if args.history < 3 or args.epochs <= 0 or args.batch <= 0 or args.hidden_size <= 0:
        raise SystemExit("history, epochs, batch, and hidden-size must be positive")
    if args.hours <= 0 or args.eval_every <= 0 or args.validation_batches <= 0 or args.prefetch_workers <= 0:
        raise SystemExit("hours, eval-every, validation-batches, and prefetch-workers must be positive")
    if args.synthetic_pretrain:
        train_synthetic(args)
        return

    sessions = collect(args.research_root, args.history, args.horizons)
    session_ids = sorted(sessions)
    all_count = sum(len(items) for items in sessions.values())
    if len(session_ids) < 2 or all_count < args.min_windows:
        raise SystemExit(
            f"Insufficient real 3D trajectory data: {len(session_ids)} sessions, {all_count} windows. "
            f"Need at least 2 sessions and {args.min_windows} windows; keep ROS depth recording on."
        )

    split = max(1, int(round(len(session_ids) * 0.2)))
    validation_ids = set(session_ids[-split:])
    train = [sample for key, samples in sessions.items() if key not in validation_ids for sample in samples]
    validation = [sample for key, samples in sessions.items() if key in validation_ids for sample in samples]
    if not train or not validation:
        raise SystemExit("session split produced an empty train or validation set")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scale = 1.0
    x_train = np.asarray([item[0] / scale for item in train], dtype=np.float32)
    y_train = np.asarray([item[1] / scale for item in train], dtype=np.float32)
    x_val = np.asarray([item[0] / scale for item in validation], dtype=np.float32)
    y_val = np.asarray([item[1] / scale for item in validation], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryGRU(hidden_size=64, outputs=len(args.horizons)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    best_state = None
    best_mae = float("inf")
    patience = 35
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            loss = torch.nn.functional.smooth_l1_loss(model(features), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
        mae = float(np.abs(predicted - y_val).mean())
        if mae < best_mae:
            best_mae, stale = mae, 0
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.pt"
    torch.save({
        "model_state": best_state,
        "history_length": args.history,
        "horizons_s": list(args.horizons),
        "position_scale_m": scale,
        "max_speed_mps": 2.0,
        "hidden_size": 64,
    }, checkpoint)
    report = {
        "checkpoint": str(checkpoint), "train_windows": len(train), "validation_windows": len(validation),
        "validation_mae_m": best_mae, "baseline_mae_m": baseline_mae(validation, args.horizons),
        "promotable": best_mae < baseline_mae(validation, args.horizons), "validation_sessions": sorted(validation_ids),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
