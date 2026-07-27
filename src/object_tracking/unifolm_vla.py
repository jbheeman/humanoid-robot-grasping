"""Unitree UnifoLM-VLA observation and action contracts.

This module intentionally contains no model or ROS imports.  The geometry
contract can therefore be tested on development machines while the heavyweight
official UnifoLM runtime remains isolated to the GB10.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


UNIFOLM_ACTION_DIM = 23
UNIFOLM_ACTION_CHUNK = 25


def rotation_matrix_to_6d(rotation: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Encode a rotation as Unitree's first-column/second-column 6D form."""

    value = np.asarray(rotation, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(value), 1.0, atol=1e-5
    ):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return tuple(float(item) for item in np.concatenate((value[:, 0], value[:, 1])))


def rotation_6d_to_matrix(values: Sequence[float]) -> np.ndarray:
    """Decode Unitree's 6D rotation with a fail-closed Gram-Schmidt step."""

    value = np.asarray(values, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation_6d must contain six finite values")
    first = value[:3]
    second = value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("rotation_6d first axis is degenerate")
    first = first / first_norm
    second = second - first * float(np.dot(first, second))
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("rotation_6d second axis is degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def transform_to_pose9(transform: Sequence[Sequence[float]]) -> tuple[float, ...]:
    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("transform must be homogeneous")
    return (
        *(float(item) for item in value[:3, 3]),
        *rotation_matrix_to_6d(value[:3, :3]),
    )


def compose_pose23(
    left_transform: Sequence[Sequence[float]],
    right_transform: Sequence[Sequence[float]],
    *,
    right_gripper: float,
    left_gripper: float,
    waist_yaw_roll_pitch: Sequence[float],
) -> tuple[float, ...]:
    """Compose Unitree's EE_R6_G1 state/action ordering.

    The gripper order is intentionally right then left.  This differs from
    Unitree's later 16D joint-controller representation.
    """

    # The public LeRobot metadata names body[12:15] yaw, roll, pitch, and
    # Unitree's converter appends that slice without reordering it.
    waist = tuple(float(item) for item in waist_yaw_roll_pitch)
    grippers = (float(right_gripper), float(left_gripper))
    if len(waist) != 3 or not all(math.isfinite(item) for item in (*grippers, *waist)):
        raise ValueError("grippers and waist_yaw_roll_pitch must be finite")
    result = (
        *transform_to_pose9(left_transform),
        *transform_to_pose9(right_transform),
        *grippers,
        *waist,
    )
    if len(result) != UNIFOLM_ACTION_DIM:  # pragma: no cover - structural invariant
        raise AssertionError("UnifoLM pose contract must contain 23 values")
    return result


@dataclass(frozen=True)
class UnifoLMWaypoint:
    left_position_m: tuple[float, float, float]
    left_rotation_6d: tuple[float, ...]
    right_position_m: tuple[float, float, float]
    right_rotation_6d: tuple[float, ...]
    right_gripper: float
    left_gripper: float
    waist_yaw_roll_pitch_rad: tuple[float, float, float]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "UnifoLMWaypoint":
        item = tuple(float(value) for value in values)
        if len(item) != UNIFOLM_ACTION_DIM or not all(math.isfinite(value) for value in item):
            raise ValueError("UnifoLM waypoint must contain 23 finite values")
        # Validate the rotation axes now so malformed model output cannot reach IK.
        rotation_6d_to_matrix(item[3:9])
        rotation_6d_to_matrix(item[12:18])
        return cls(
            left_position_m=item[0:3],
            left_rotation_6d=item[3:9],
            right_position_m=item[9:12],
            right_rotation_6d=item[12:18],
            right_gripper=item[18],
            left_gripper=item[19],
            waist_yaw_roll_pitch_rad=item[20:23],
        )

    def right_transform(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = rotation_6d_to_matrix(self.right_rotation_6d)
        result[:3, 3] = self.right_position_m
        return result


def parse_action_chunk(values: Sequence[Sequence[float]]) -> tuple[UnifoLMWaypoint, ...]:
    chunk = tuple(UnifoLMWaypoint.from_values(item) for item in values)
    if not chunk or len(chunk) > UNIFOLM_ACTION_CHUNK:
        raise ValueError(f"UnifoLM action chunk must contain 1-{UNIFOLM_ACTION_CHUNK} waypoints")
    return chunk
