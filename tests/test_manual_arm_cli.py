import pytest

from object_tracking.manual_arm_cli import (
    RemoteArmError,
    _signed_progress,
    _validate_args,
    build_parser,
)


def test_move_defaults_to_guarded_right_shoulder_cycle() -> None:
    args = build_parser().parse_args(["move"])
    _validate_args(args)
    assert args.side == "right"
    assert args.delta == 0.05
    assert args.duration == 2.0
    assert args.no_return is False


def test_move_rejects_delta_beyond_robot_side_step_limit() -> None:
    args = build_parser().parse_args(["move", "--delta", "0.051"])
    with pytest.raises(RemoteArmError, match="0.05"):
        _validate_args(args)


def test_signed_progress_rejects_motion_opposite_the_requested_direction() -> None:
    assert _signed_progress(0.1, 0.14, 0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.06, -0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.14, -0.05) == pytest.approx(-0.04)
