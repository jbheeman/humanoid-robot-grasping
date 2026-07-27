import numpy as np
import pytest

from object_tracking.unifolm_relative_actions import (
    pose17_to_pose23,
    reconstruct_anchored_pose23,
    to_anchored_relative_pose23,
)
from object_tracking.unifolm_vla import (
    compose_pose23,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)


def _transform(position: tuple[float, float, float], yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    transform[:3, 3] = position
    return transform


def _pose(offset: float) -> np.ndarray:
    return np.asarray(
        compose_pose23(
            _transform((0.1 + offset, 0.2, 0.3), 0.2 + offset),
            _transform((0.3 + offset, -0.1, 0.2), -0.3 + offset),
            right_gripper=0.2 + offset,
            left_gripper=0.4,
            waist_yaw_roll_pitch=(0.1 + offset, 0.0, -0.1),
        ),
        dtype=np.float32,
    )


def test_anchored_relative_pose23_round_trips_chunk() -> None:
    anchor = _pose(0.0)
    targets = np.stack((_pose(0.01), _pose(0.04), _pose(-0.02)))

    relative = to_anchored_relative_pose23(anchor, targets)
    reconstructed = reconstruct_anchored_pose23(anchor, relative)

    np.testing.assert_allclose(reconstructed, targets, atol=1e-6)


def test_translation_delta_remains_in_torso_axes() -> None:
    anchor = _pose(0.0)
    target = anchor.copy()
    target[9:12] += (0.05, -0.02, 0.01)

    relative = to_anchored_relative_pose23(anchor, target)

    np.testing.assert_allclose(relative[9:12], (0.05, -0.02, 0.01), atol=1e-7)


def test_identical_pose_encodes_identity_relative_rotations() -> None:
    anchor = _pose(0.0)

    relative = to_anchored_relative_pose23(anchor, anchor)

    np.testing.assert_allclose(relative[[0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22]], 0.0)
    np.testing.assert_allclose(
        relative[3:9],
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        relative[12:18],
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "axis",
    (
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    ),
)
@pytest.mark.parametrize("angle", (-np.pi / 2, np.pi / 2))
def test_rotation_column_6d_round_trip(axis: np.ndarray, angle: float) -> None:
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )

    decoded = rotation_6d_to_matrix(rotation_matrix_to_6d(rotation))

    np.testing.assert_allclose(decoded, rotation, atol=1e-7)


def test_pose17_conversion_matches_column_rotation_convention() -> None:
    pose17 = np.zeros(17, dtype=np.float32)
    pose17[3:6] = (0.2, -0.3, 0.4)
    pose23 = pose17_to_pose23(pose17)

    rotation = rotation_6d_to_matrix(pose23[3:9])

    assert np.isclose(np.linalg.det(rotation), 1.0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)


@pytest.mark.parametrize(
    ("anchor", "target"),
    (
        (np.zeros(22), np.zeros(23)),
        (np.zeros(23), np.zeros((2, 22))),
        (np.full(23, np.nan), np.zeros(23)),
    ),
)
def test_relative_pose23_rejects_invalid_contract(
    anchor: np.ndarray,
    target: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        to_anchored_relative_pose23(anchor, target)
