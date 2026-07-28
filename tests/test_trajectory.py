from __future__ import annotations

import pytest

from object_tracking.arm_tracking.trajectory import minimum_ruckig_duration_s


LIMITS = {
    "maximum_velocity": 0.5,
    "maximum_acceleration": 2.0,
    "maximum_jerk": 20.0,
}


def duration(target: tuple[float, ...], **changes: object) -> float:
    values: dict[str, object] = {
        "current_position": (0.0,) * 7,
        "target_position": target,
        **LIMITS,
    }
    values.update(changes)
    return minimum_ruckig_duration_s(**values)  # type: ignore[arg-type]


def test_zero_motion_has_zero_duration() -> None:
    assert duration((0.0,) * 7) == pytest.approx(0.0)


def test_farther_joint_target_takes_longer() -> None:
    near = duration((0.05,) + (0.0,) * 6)
    far = duration((0.30,) + (0.0,) * 6)

    assert near > 0.0
    assert far > near


def test_measured_velocity_changes_reachable_duration() -> None:
    stationary = duration((0.30,) + (0.0,) * 6)
    already_moving = duration(
        (0.30,) + (0.0,) * 6,
        current_velocity=(0.25,) + (0.0,) * 6,
    )

    assert already_moving < stationary


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_position": (0.0,) * 6}, "target_position"),
        ({"current_velocity": (float("nan"),) + (0.0,) * 6}, "current_velocity"),
        ({"maximum_velocity": 0.0}, "maximum_velocity"),
        ({"maximum_acceleration": (1.0,) * 6}, "maximum_acceleration"),
    ],
)
def test_invalid_inputs_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        duration((0.1,) * 7, **changes)
