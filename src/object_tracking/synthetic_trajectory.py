"""Procedural, bounded trajectories for forecaster pretraining.

The generator deliberately produces only position histories and future offsets;
it does not pretend to be labelled robot telemetry.  It is suitable for a
cheap pretraining prior before a checkpoint is evaluated on recorded sessions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .trajectory_forecaster import trajectory_features


def synthetic_batch(
    rng: np.random.Generator,
    *,
    batch_size: int,
    history: int,
    horizons_s: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create one noisy, irregularly sampled batch of smooth 3D motion.

    Positions are in a conservative torso-relative workspace.  A missing
    observation is represented by a doubled sample interval, which remains
    below the runtime forecaster's 0.20-second reset gap.
    """
    horizons = np.asarray(tuple(horizons_s), dtype=np.float32)
    if batch_size <= 0 or history < 3 or horizons.ndim != 1 or np.any(horizons <= 0):
        raise ValueError("batch_size, history, and horizons must be positive")

    nominal_dt = rng.uniform(1.0 / 15.0, 1.0 / 10.0, size=(batch_size, history - 1))
    nominal_dt *= rng.uniform(0.88, 1.12, size=nominal_dt.shape)
    missing = rng.random(nominal_dt.shape) < 0.08
    nominal_dt[missing] *= 1.75
    times = np.concatenate(
        (-np.cumsum(nominal_dt[:, ::-1], axis=1)[:, ::-1], np.zeros((batch_size, 1))), axis=1
    ).astype(np.float32)

    # Smooth acceleration plus a low-frequency component covers throws,
    # reversals, and hand-carried motion without generating impossible speeds.
    origin = rng.uniform([-0.35, -0.55, 0.20], [0.65, 0.25, 1.20], size=(batch_size, 3))
    velocity = rng.normal(0.0, 0.38, size=(batch_size, 3)).clip(-0.9, 0.9)
    acceleration = rng.normal(0.0, 1.15, size=(batch_size, 3)).clip(-2.5, 2.5)
    amplitude = rng.uniform(0.0, 0.055, size=(batch_size, 3))
    frequency = rng.uniform(1.0, 5.0, size=(batch_size, 1))
    phase = rng.uniform(-np.pi, np.pi, size=(batch_size, 3))

    def displacement(t: np.ndarray) -> np.ndarray:
        values = t[..., None]
        harmonic = amplitude[:, None, :] * (
            np.sin(frequency[:, None, :] * values + phase[:, None, :])
            - np.sin(phase[:, None, :])
        )
        return (
            velocity[:, None, :] * values
            + 0.5 * acceleration[:, None, :] * values**2
            + harmonic
        )

    clean_history = origin[:, None, :] + displacement(times)
    # 3–12 mm measurement noise approximates depth projection and detector
    # center jitter while keeping target labels noise-free.
    noisy_history = clean_history + rng.normal(0.0, rng.uniform(0.003, 0.012, (batch_size, 1, 1)), clean_history.shape)
    # This is the batched equivalent of trajectory_features.  Avoiding a
    # Python loop is important for the large pretraining batches used on GB10.
    features = np.empty((batch_size, history, 6), dtype=np.float32)
    features[:, :, :3] = noisy_history - noisy_history[:, -1:, :]
    estimated_velocity = np.empty_like(noisy_history)
    estimated_velocity[:, 1:] = np.diff(noisy_history, axis=1) / np.diff(times, axis=1)[:, :, None]
    estimated_velocity[:, 0] = estimated_velocity[:, 1]
    features[:, :, 3:] = estimated_velocity
    target_times = np.broadcast_to(horizons[None, :], (batch_size, len(horizons)))
    targets = displacement(target_times).astype(np.float32)
    return features, targets
