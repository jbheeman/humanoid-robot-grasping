"""Anchored relative-action contract for UniFoLM's 23D G1 EE representation.

Every action in a chunk is expressed relative to the newest proprioceptive
observation that produced that chunk. Translation deltas remain in torso axes;
rotation deltas are local compositions ``R_anchor.T @ R_target``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from object_tracking.unifolm_vla import (
    UNIFOLM_ACTION_DIM,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)


RELATIVE_POSE23_V1 = "anchored_relative_pose23_v1"
_POSE_BASES = (0, 9)


def _pose23(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != UNIFOLM_ACTION_DIM:
        raise ValueError(f"{name} must end in {UNIFOLM_ACTION_DIM} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rotation_batch(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, 6)
    matrices = np.stack([rotation_6d_to_matrix(item) for item in flat])
    return matrices.reshape(*values.shape[:-1], 3, 3)


def _rotation_6d_batch(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, 3, 3)
    encoded = np.asarray(
        [rotation_matrix_to_6d(item) for item in flat],
        dtype=np.float64,
    )
    return encoded.reshape(*values.shape[:-2], 6)


def pose17_to_pose23(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert canonical xyz+rpy Pose17 values to Unitree's column-6D Pose23."""

    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim < 1 or poses.shape[-1] != 17:
        raise ValueError("Pose17 values must end in 17 values")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Pose17 values must contain only finite values")

    def rotation6d(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        first = np.stack((cy * cp, sy * cp, -sp), axis=-1)
        second = np.stack(
            (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
            axis=-1,
        )
        return np.concatenate((first, second), axis=-1)

    left = np.concatenate((poses[..., 0:3], rotation6d(poses[..., 3:6])), axis=-1)
    right = np.concatenate((poses[..., 6:9], rotation6d(poses[..., 9:12])), axis=-1)
    return np.concatenate((left, right, poses[..., 12:17]), axis=-1).astype(np.float32)


def to_anchored_relative_pose23(
    anchor: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Encode absolute targets relative to one broadcastable Pose23 anchor."""

    anchor_values, target_values = np.broadcast_arrays(
        _pose23(anchor, "anchor"),
        _pose23(targets, "targets"),
    )
    result = np.empty_like(target_values)
    for base in _POSE_BASES:
        result[..., base : base + 3] = (
            target_values[..., base : base + 3]
            - anchor_values[..., base : base + 3]
        )
        anchor_rotation = _rotation_batch(anchor_values[..., base + 3 : base + 9])
        target_rotation = _rotation_batch(target_values[..., base + 3 : base + 9])
        relative_rotation = np.swapaxes(anchor_rotation, -1, -2) @ target_rotation
        result[..., base + 3 : base + 9] = _rotation_6d_batch(relative_rotation)
    result[..., 18:23] = target_values[..., 18:23] - anchor_values[..., 18:23]
    return result.astype(np.float32)


def reconstruct_anchored_pose23(
    anchor: Sequence[float] | np.ndarray,
    relative_actions: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Reconstruct absolute Pose23 targets from anchored relative actions."""

    anchor_values, relative_values = np.broadcast_arrays(
        _pose23(anchor, "anchor"),
        _pose23(relative_actions, "relative_actions"),
    )
    result = np.empty_like(relative_values)
    for base in _POSE_BASES:
        result[..., base : base + 3] = (
            anchor_values[..., base : base + 3]
            + relative_values[..., base : base + 3]
        )
        anchor_rotation = _rotation_batch(anchor_values[..., base + 3 : base + 9])
        relative_rotation = _rotation_batch(relative_values[..., base + 3 : base + 9])
        target_rotation = anchor_rotation @ relative_rotation
        result[..., base + 3 : base + 9] = _rotation_6d_batch(target_rotation)
    result[..., 18:23] = anchor_values[..., 18:23] + relative_values[..., 18:23]
    return result.astype(np.float32)
