"""Deterministic plush-motion curriculum for Isaac episode randomization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np


@dataclass(frozen=True)
class TrajectorySpec:
    stage: str
    seed: int
    start_xyz: tuple[float, float, float]
    velocity_xyz: tuple[float, float, float]
    lateral_amplitude: float = 0.0
    lateral_frequency_hz: float = 0.0

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(self.start_xyz, dtype=np.float32)
        velocity = np.asarray(self.velocity_xyz, dtype=np.float32)
        position = start + velocity * t
        if self.lateral_amplitude:
            omega = 2.0 * math.pi * self.lateral_frequency_hz
            position[1] += self.lateral_amplitude * math.sin(omega * t)
            velocity = velocity.copy()
            velocity[1] += self.lateral_amplitude * omega * math.cos(omega * t)
        return position.astype(np.float32), velocity.astype(np.float32)


def sample_trajectory(stage: str, seed: int) -> TrajectorySpec:
    rng = random.Random(seed)
    start = (rng.uniform(0.45, 0.65), rng.uniform(-0.18, 0.18), rng.uniform(0.78, 0.88))
    if stage == "static_grasp":
        return TrajectorySpec(stage, seed, start, (0.0, 0.0, 0.0))
    if stage == "slow_linear":
        speed = rng.uniform(0.03, 0.08)
        return TrajectorySpec(stage, seed, start, (0.0, rng.choice((-speed, speed)), 0.0))
    if stage == "varied_motion":
        speed = rng.uniform(0.03, 0.15)
        return TrajectorySpec(
            stage,
            seed,
            start,
            (rng.uniform(-0.02, 0.02), rng.choice((-speed, speed)), 0.0),
            lateral_amplitude=rng.uniform(0.01, 0.05),
            lateral_frequency_hz=rng.uniform(0.1, 0.4),
        )
    raise ValueError(f"unknown curriculum stage: {stage}")
