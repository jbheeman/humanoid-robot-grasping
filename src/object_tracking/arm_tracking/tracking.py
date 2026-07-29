"""Lightweight 3D filtering, prediction, and sticky target selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _point(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("position must contain three finite values")
    return result


@dataclass(frozen=True)
class TrackedPosition:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    timestamp_s: float

    def predict(self, horizon_s: float = 0.150) -> np.ndarray:
        if horizon_s < 0:
            raise ValueError("horizon_s must be non-negative")
        return self.position_m + self.velocity_mps * horizon_s


class PositionVelocityFilter:
    """Robust alpha-beta filter with bounded innovation and acceleration."""

    def __init__(
        self,
        *,
        position_gain: float = 0.45,
        velocity_gain: float = 0.12,
        reset_gap_s: float = 0.5,
        max_speed_mps: float = 0.75,
        max_acceleration_mps2: float = 2.0,
        max_innovation_m: float = 0.03,
        velocity_damping: float = 0.90,
        max_prediction_displacement_m: float = 0.06,
    ) -> None:
        if not 0 < position_gain <= 1 or not 0 <= velocity_gain <= 1:
            raise ValueError("filter gains must be within [0, 1]")
        if not 0 < velocity_damping <= 1:
            raise ValueError("velocity damping must be within (0, 1]")
        if any(
            value <= 0
            for value in (
                reset_gap_s,
                max_speed_mps,
                max_acceleration_mps2,
                max_innovation_m,
                max_prediction_displacement_m,
            )
        ):
            raise ValueError("filter limits must be positive")
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.reset_gap_s = reset_gap_s
        self.max_speed_mps = max_speed_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.max_innovation_m = max_innovation_m
        self.velocity_damping = velocity_damping
        self.max_prediction_displacement_m = max_prediction_displacement_m
        self._state: TrackedPosition | None = None
        self._consecutive_observations = 0
        self._last_residual_m = 0.0

    @property
    def state(self) -> TrackedPosition | None:
        return self._state

    @property
    def consecutive_observations(self) -> int:
        return self._consecutive_observations

    @property
    def last_residual_m(self) -> float:
        return self._last_residual_m

    def reset(self) -> None:
        self._state = None
        self._consecutive_observations = 0
        self._last_residual_m = 0.0

    def update(self, measurement_m: Iterable[float], timestamp_s: float) -> TrackedPosition:
        measurement = _point(measurement_m)
        if not np.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        previous = self._state
        if previous is None or timestamp_s - previous.timestamp_s > self.reset_gap_s:
            self._state = TrackedPosition(measurement, np.zeros(3), timestamp_s)
            self._consecutive_observations = 1
            self._last_residual_m = 0.0
            return self._state
        dt = timestamp_s - previous.timestamp_s
        if dt <= 0:
            raise ValueError("filter timestamps must be strictly increasing")
        predicted = previous.position_m + previous.velocity_mps * dt
        residual = measurement - predicted
        residual_norm = float(np.linalg.norm(residual))
        self._last_residual_m = residual_norm
        if residual_norm > self.max_innovation_m:
            residual = residual * (self.max_innovation_m / residual_norm)
        position = predicted + self.position_gain * residual
        candidate_velocity = (
            previous.velocity_mps * self.velocity_damping
            + self.velocity_gain * residual / dt
        )
        velocity_delta = candidate_velocity - previous.velocity_mps
        velocity_delta_norm = float(np.linalg.norm(velocity_delta))
        maximum_velocity_delta = self.max_acceleration_mps2 * dt
        if velocity_delta_norm > maximum_velocity_delta:
            velocity_delta = velocity_delta * (
                maximum_velocity_delta / velocity_delta_norm
            )
        velocity = previous.velocity_mps + velocity_delta
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed_mps:
            velocity = velocity * (self.max_speed_mps / speed)
        self._state = TrackedPosition(position, velocity, timestamp_s)
        self._consecutive_observations += 1
        return self._state

    def predict(self, horizon_s: float = 0.150) -> np.ndarray | None:
        if horizon_s < 0:
            raise ValueError("horizon_s must be non-negative")
        if self._state is None:
            return None
        displacement = self._state.velocity_mps * horizon_s
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm > self.max_prediction_displacement_m:
            displacement = displacement * (
                self.max_prediction_displacement_m / displacement_norm
            )
        return self._state.position_m + displacement


@dataclass(frozen=True)
class TargetCandidate:
    track_id: int
    confidence: float
    position_m: np.ndarray
    depth_certain: bool = True

    def __post_init__(self) -> None:
        if self.track_id < 0 or not 0 <= self.confidence <= 1:
            raise ValueError("candidate track ID and confidence are invalid")
        object.__setattr__(self, "position_m", _point(self.position_m))


class StickyTargetSelector:
    """Prefer continuity, but release a target after a bounded loss interval."""

    def __init__(self, *, loss_timeout_s: float = 0.35, min_confidence: float = 0.25) -> None:
        if loss_timeout_s <= 0 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid target selector thresholds")
        self.loss_timeout_s = loss_timeout_s
        self.min_confidence = min_confidence
        self.active_track_id: int | None = None
        self.last_seen_s: float | None = None

    def update(
        self, candidates: Iterable[TargetCandidate], timestamp_s: float
    ) -> TargetCandidate | None:
        if not np.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        eligible = [
            candidate
            for candidate in candidates
            if candidate.depth_certain and candidate.confidence >= self.min_confidence
        ]
        current = next((item for item in eligible if item.track_id == self.active_track_id), None)
        if current is not None:
            self.last_seen_s = timestamp_s
            return current
        if (
            self.active_track_id is not None
            and self.last_seen_s is not None
            and timestamp_s - self.last_seen_s <= self.loss_timeout_s
        ):
            return None
        if not eligible:
            self.active_track_id = None
            self.last_seen_s = None
            return None
        selected = max(eligible, key=lambda item: (item.confidence, -item.track_id))
        self.active_track_id = selected.track_id
        self.last_seen_s = timestamp_s
        return selected

    def is_lost(self, timestamp_s: float) -> bool:
        if not np.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        return bool(
            self.active_track_id is not None
            and self.last_seen_s is not None
            and timestamp_s - self.last_seen_s > self.loss_timeout_s
        )
