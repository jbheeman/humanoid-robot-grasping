"""Canonical sample contract shared by Isaac capture and dataset conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


CAMERA_NAMES = (
    "cam_left_high",
    "cam_right_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


@dataclass(frozen=True)
class DatasetContract:
    fps: int = 30
    image_height: int = 480
    image_width: int = 640
    qpos_dim: int = 19
    ee_dim: int = 17
    unifolm_ee_r6_dim: int = 23
    action_chunk: int = 25


@dataclass
class FrameSample:
    """One synchronized observation/action pair.

    Actions are absolute expert targets at the current timestamp. Gripper
    ordering follows Unitree's official converter: right before left.
    """

    timestamp: float
    images: Mapping[str, np.ndarray]
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    ee_qpos: np.ndarray
    ee_action: np.ndarray
    plush_position: np.ndarray
    plush_velocity: np.ndarray
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    tracking_valid: bool = True
    grasped: bool = False
    safety_event: bool = False

    def validate(self, contract: DatasetContract = DatasetContract()) -> None:
        expected_vectors = {
            "qpos": (self.qpos, contract.qpos_dim),
            "qvel": (self.qvel, contract.qpos_dim),
            "action": (self.action, contract.qpos_dim),
            "ee_qpos": (self.ee_qpos, contract.ee_dim),
            "ee_action": (self.ee_action, contract.ee_dim),
            "plush_position": (self.plush_position, 3),
            "plush_velocity": (self.plush_velocity, 3),
            "contact_force": (self.contact_force, 6),
        }
        for name, (value, width) in expected_vectors.items():
            array = np.asarray(value)
            if array.shape != (width,):
                raise ValueError(f"{name} must have shape ({width},), got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")

        missing = set(CAMERA_NAMES) - set(self.images)
        if missing:
            raise ValueError(f"missing cameras: {sorted(missing)}")
        for name in CAMERA_NAMES:
            image = np.asarray(self.images[name])
            expected = (contract.image_height, contract.image_width, 3)
            if image.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {image.shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8, got {image.dtype}")


def rpy_batch_to_rotation_6d(rpy: np.ndarray) -> np.ndarray:
    """Convert [..., roll/pitch/yaw] to the first two rotation-matrix columns."""

    rpy = np.asarray(rpy, dtype=np.float32)
    if rpy.shape[-1] != 3:
        raise ValueError(f"expected final dimension 3, got {rpy.shape}")
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    col1 = np.stack((cy * cp, sy * cp, -sp), axis=-1)
    col2 = np.stack(
        (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        axis=-1,
    )
    return np.concatenate((col1, col2), axis=-1).astype(np.float32)


def ee17_to_unifolm23(values: np.ndarray) -> np.ndarray:
    """Match UniFoLM's official `batch_pose17_to_pose23` conversion exactly."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 17:
        raise ValueError(f"expected final dimension 17, got {values.shape}")
    left = np.concatenate((values[..., 0:3], rpy_batch_to_rotation_6d(values[..., 3:6])), axis=-1)
    right = np.concatenate((values[..., 6:9], rpy_batch_to_rotation_6d(values[..., 9:12])), axis=-1)
    return np.concatenate((left, right, values[..., 12:17]), axis=-1).astype(np.float32)

