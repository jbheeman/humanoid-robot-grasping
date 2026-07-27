"""Offline future-position evaluation for bunny detection tracks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .arm_tracking.tracking import PositionVelocityFilter


@dataclass(frozen=True)
class TrackObservation:
    timestamp_s: float
    track_id: int
    center_xy: np.ndarray
    depth_m: float | None
    confidence: float

    @property
    def state_m(self) -> np.ndarray:
        return np.array(
            [self.center_xy[0], self.center_xy[1], 0.0 if self.depth_m is None else self.depth_m],
            dtype=np.float64,
        )


def load_observations(path: str | Path) -> list[TrackObservation]:
    """Load JSONL records with timestamp, track_id, center_xy, and optional depth_m."""
    observations: list[TrackObservation] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        center = np.asarray(value["center_xy"], dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center_xy must contain two finite values")
        depth = value.get("depth_m")
        observations.append(
            TrackObservation(
                timestamp_s=float(value["timestamp_s"]),
                track_id=int(value["track_id"]),
                center_xy=center,
                depth_m=None if depth is None else float(depth),
                confidence=float(value.get("confidence", 1.0)),
            )
        )
    return sorted(observations, key=lambda value: (value.track_id, value.timestamp_s))


def _future_sample(
    observations: list[TrackObservation], index: int, horizon_s: float
) -> TrackObservation | None:
    target_time = observations[index].timestamp_s + horizon_s
    return next((item for item in observations[index + 1 :] if item.timestamp_s >= target_time), None)


def evaluate_future_positions(
    observations: Iterable[TrackObservation],
    *,
    horizons_s: tuple[float, ...] = (0.1, 0.2, 0.5),
    min_confidence: float = 0.25,
) -> dict[str, Any]:
    """Score alpha-beta future predictions against later observed track positions."""
    grouped: dict[int, list[TrackObservation]] = {}
    for observation in observations:
        if observation.confidence >= min_confidence:
            grouped.setdefault(observation.track_id, []).append(observation)
    report: dict[str, Any] = {
        "schema_version": 1,
        "min_confidence": min_confidence,
        "horizons": {},
        "tracks": len(grouped),
    }
    for horizon in horizons_s:
        pixel_errors: list[float] = []
        depth_errors: list[float] = []
        three_d_errors: list[float] = []
        for track in grouped.values():
            tracker = PositionVelocityFilter(
                position_gain=0.85, velocity_gain=0.70, reset_gap_s=0.5, max_speed_mps=5000.0
            )
            for index, observation in enumerate(track):
                state = tracker.update(observation.state_m, observation.timestamp_s)
                future = _future_sample(track, index, horizon)
                if future is None:
                    continue
                predicted = state.predict(horizon)
                pixel_errors.append(float(np.linalg.norm(predicted[:2] - future.center_xy)))
                if observation.depth_m is not None and future.depth_m is not None:
                    depth_errors.append(float(abs(predicted[2] - future.depth_m)))
                    three_d_errors.append(float(np.linalg.norm(predicted - future.state_m)))
        report["horizons"][f"{int(horizon * 1000)}ms"] = {
            "samples": len(pixel_errors),
            "pixel_median": None if not pixel_errors else float(np.median(pixel_errors)),
            "pixel_p95": None if not pixel_errors else float(np.percentile(pixel_errors, 95)),
            "depth_median_m": None if not depth_errors else float(np.median(depth_errors)),
            "depth_p95_m": None if not depth_errors else float(np.percentile(depth_errors, 95)),
            "three_d_median": None if not three_d_errors else float(np.median(three_d_errors)),
            "three_d_p95": None if not three_d_errors else float(np.percentile(three_d_errors, 95)),
        }
    return report


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "future_prediction_report.json"
    markdown_path = directory / "future_prediction_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = [
        "| Horizon | Samples | Median px | P95 px | Median 3D m | P95 3D m |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon, metrics in report["horizons"].items():
        display = {key: "n/a" if value is None else f"{value:.4f}" for key, value in metrics.items()}
        rows.append("| {horizon} | {samples} | {pixel_median} | {pixel_p95} | {three_d_median} | {three_d_p95} |".format(horizon=horizon, **display))
    markdown_path.write_text("# Bunny future-position replay\n\n" + "\n".join(rows) + "\n")
    return json_path, markdown_path
