import pytest

from object_tracking.manual_arm_cli import (
    RemoteArmError,
    _arming_failure,
    _arming_status_summary,
    _guarded_offsets,
    _manual_deltas,
    _signed_progress,
    _validate_args,
    build_parser,
)


def test_arming_status_reports_the_safety_reason_without_joint_dump() -> None:
    status = {
        "state": "HOLDING",
        "session_id": "session-a",
        "weight": 0.42,
        "fault_reason": "runtime_gate:lowstate_stale",
        "hold_reason": "runtime_gate:lowstate_stale",
        "measured_arm_q": [0.0] * 14,
    }
    failure = _arming_failure(status, "session-a", session_seen=True)
    assert failure is not None
    assert "runtime_gate:lowstate_stale" in failure
    assert "state='HOLDING'" in failure
    assert "measured_arm_q" not in failure


def test_arming_failure_survives_robot_session_cleanup() -> None:
    status = {
        "state": "FAULT",
        "session_id": None,
        "weight": 0.0,
        "fault_reason": "following_error:0.0312",
        "hold_reason": "following_error:0.0312",
    }
    failure = _arming_failure(status, "session-a", session_seen=True)
    assert failure is not None
    assert "following_error:0.0312" in failure
    assert _arming_failure(status, "session-a", session_seen=False) is None


def test_arming_timeout_summary_includes_gate_values() -> None:
    summary = _arming_status_summary(
        {
            "state": "ARMING",
            "weight": 0.2,
            "motion_mode_verified": False,
            "robot_state_age_ms": 300.0,
        }
    )
    assert "motion_mode_verified=False" in summary
    assert "robot_state_age_ms=300.0" in summary


def test_move_defaults_to_guarded_right_shoulder_cycle() -> None:
    args = build_parser().parse_args(["move"])
    _validate_args(args)
    assert args.side == "right"
    assert args.delta == 0.05
    assert args.duration == 2.0
    assert args.no_return is False


def test_manual_multi_joint_deltas_parse_together() -> None:
    args = build_parser().parse_args(
        [
            "move",
            "--side",
            "right",
            "--joint-delta",
            "right_shoulder_pitch_joint=0.15",
            "--joint-delta",
            "right_elbow_joint=-0.10",
        ]
    )
    _validate_args(args)
    assert _manual_deltas(
        args,
        (
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
        ),
    ) == {
        "right_shoulder_pitch_joint": 0.15,
        "right_elbow_joint": -0.10,
    }


def test_ik_accepts_small_pelvis_frame_offset() -> None:
    args = build_parser().parse_args(["ik", "--dz", "0.01"])
    _validate_args(args)
    assert args.dx == 0.0
    assert args.dy == 0.0
    assert args.dz == 0.01


def test_ik_rejects_zero_or_large_offset() -> None:
    with pytest.raises(RemoteArmError, match="offset norm"):
        _validate_args(build_parser().parse_args(["ik"]))
    with pytest.raises(RemoteArmError, match="offset norm"):
        _validate_args(build_parser().parse_args(["ik", "--dx", "0.051"]))


def test_move_allows_visible_delta_split_into_guarded_steps() -> None:
    args = build_parser().parse_args(["move", "--delta", "0.15"])
    _validate_args(args)
    assert _guarded_offsets(args.delta) == pytest.approx([0.05, 0.10, 0.15])


def test_move_rejects_delta_beyond_manual_total_limit() -> None:
    args = build_parser().parse_args(["move", "--delta", "0.201"])
    with pytest.raises(RemoteArmError, match="0.20"):
        _validate_args(args)


def test_guarded_offsets_keep_each_target_step_bounded() -> None:
    offsets = [0.0, *_guarded_offsets(-0.16)]
    assert offsets[-1] == pytest.approx(-0.16)
    assert max(abs(end - start) for start, end in zip(offsets, offsets[1:])) <= 0.05


def test_signed_progress_rejects_motion_opposite_the_requested_direction() -> None:
    assert _signed_progress(0.1, 0.14, 0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.06, -0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.14, -0.05) == pytest.approx(-0.04)
