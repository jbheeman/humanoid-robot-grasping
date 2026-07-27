"""Optional learned 3D trajectory forecasting with a safe classical fallback.

The trained model predicts displacement at a small set of fixed horizons.  It is
deliberately optional: callers must retain a deterministic filter when there is
not enough recent, valid 3D history or a learned prediction is implausible.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np


def trajectory_features(times_s: Iterable[float], positions_m: Iterable[Iterable[float]]) -> np.ndarray:
    """Return translation-invariant position/velocity features for a trajectory."""
    times = np.asarray(tuple(times_s), dtype=np.float32)
    positions = np.asarray(tuple(positions_m), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(times) != len(positions):
        raise ValueError("times and positions must describe an Nx3 trajectory")
    if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("trajectory needs at least two finite observations")
    delta_t = np.diff(times)
    if np.any(delta_t <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    velocity = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / delta_t[:, None]
    velocity[0] = velocity[1]
    return np.concatenate((positions - positions[-1], velocity), axis=1)


class LearnedTrajectoryForecaster:
    """Load a compact Torch GRU checkpoint and expose validated predictions."""

    def __init__(self, model_path: str | Path, *, max_gap_s: float = 0.20) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on vision hosts.
            raise RuntimeError("Torch is required to load a trajectory model") from exc
        from .trajectory_training import TrajectoryGRU

        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
        self.history_length = int(checkpoint["history_length"])
        self.horizons_s = tuple(float(value) for value in checkpoint["horizons_s"])
        self.position_scale_m = float(checkpoint["position_scale_m"])
        self.max_speed_mps = float(checkpoint.get("max_speed_mps", 2.0))
        self.max_gap_s = float(max_gap_s)
        self.torch = torch
        self.model = TrajectoryGRU(hidden_size=int(checkpoint["hidden_size"]), outputs=len(self.horizons_s))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        self.history.clear()

    def update(self, position_m: Iterable[float], timestamp_s: float) -> None:
        position = np.asarray(tuple(position_m), dtype=np.float32)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            self.reset()
            return
        if self.history and (timestamp_s <= self.history[-1][0] or timestamp_s - self.history[-1][0] > self.max_gap_s):
            self.reset()
        self.history.append((float(timestamp_s), position))

    def predict(self, horizon_s: float) -> np.ndarray | None:
        if len(self.history) != self.history_length:
            return None
        index = min(range(len(self.horizons_s)), key=lambda item: abs(self.horizons_s[item] - horizon_s))
        if abs(self.horizons_s[index] - horizon_s) > 0.035:
            return None
        times = [item[0] for item in self.history]
        positions = [item[1] for item in self.history]
        try:
            features = trajectory_features(times, positions) / self.position_scale_m
        except ValueError:
            self.reset()
            return None
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(features).unsqueeze(0))[0, index].numpy()
        prediction = positions[-1] + output * self.position_scale_m
        maximum_displacement = self.max_speed_mps * horizon_s * 1.25
        if not np.all(np.isfinite(prediction)) or np.linalg.norm(prediction - positions[-1]) > maximum_displacement:
            return None
        return prediction.astype(np.float64)
