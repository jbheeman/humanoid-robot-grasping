import pytest

from object_tracking.manual_arm_cli import RemoteArmError, _validate_args, build_parser


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
