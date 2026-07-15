import json
import math

import pytest

from object_tracking.arm_tracking.joints import RIGHT_ARM_JOINT_NAMES
from object_tracking.manual_arm_cli import (
    RemoteArmError,
    MAX_IK_WAYPOINT_DISTANCE_M,
    MAX_IK_WAYPOINT_JOINT_DELTA_RAD,
    MAX_IK_CARTESIAN_OFFSET_M,
    MAX_MANUAL_TOTAL_DELTA_RAD,
    _arming_failure,
    _arming_status_summary,
    _fetch_vision_target,
    _guarded_offsets,
    _guarded_joint_path,
    _manual_deltas,
    _signed_progress,
    _validate_args,
    _write_motion_trace,
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
    assert args.stay is False
    assert MAX_IK_WAYPOINT_DISTANCE_M == pytest.approx(0.01)
    assert MAX_IK_WAYPOINT_JOINT_DELTA_RAD > MAX_MANUAL_TOTAL_DELTA_RAD


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


def test_manual_seven_joint_vector_uses_canonical_arm_order() -> None:
    args = build_parser().parse_args(
        [
            "move",
            "--side",
            "right",
            "--arm-deltas",
            "0.10",
            "-0.05",
            "0",
            "0.08",
            "0",
            "0.02",
            "0",
        ]
    )
    _validate_args(args)
    assert _manual_deltas(args, RIGHT_ARM_JOINT_NAMES) == {
        "right_shoulder_pitch_joint": 0.10,
        "right_shoulder_roll_joint": -0.05,
        "right_elbow_joint": 0.08,
        "right_wrist_pitch_joint": 0.02,
    }


def test_manual_seven_joint_vector_rejects_empty_or_mixed_selection() -> None:
    with pytest.raises(RemoteArmError, match="at least one"):
        _validate_args(build_parser().parse_args(["move", "--arm-deltas", *(["0"] * 7)]))
    with pytest.raises(RemoteArmError, match="only one"):
        _validate_args(
            build_parser().parse_args(
                [
                    "move",
                    "--joint",
                    "right_elbow_joint",
                    "--arm-deltas",
                    "0.1",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                ]
            )
        )


def test_ik_accepts_small_pelvis_frame_offset() -> None:
    args = build_parser().parse_args(
        ["ik", "--dz", "0.01", "--trace-output", "runs/arm/trace.json"]
    )
    _validate_args(args)
    assert args.dx == 0.0
    assert args.dy == 0.0
    assert args.dz == 0.01
    assert args.trace_output.as_posix() == "runs/arm/trace.json"


def test_ik_accepts_visible_full_arm_offset() -> None:
    args = build_parser().parse_args(["ik", "--dy", "-0.075", "--dz", "0.1"])
    _validate_args(args)
    assert math.hypot(args.dy, args.dz) == pytest.approx(0.125)
    assert MAX_IK_CARTESIAN_OFFSET_M == pytest.approx(0.5)


def test_ik_rejects_zero_or_large_offset() -> None:
    with pytest.raises(RemoteArmError, match="offset norm"):
        _validate_args(build_parser().parse_args(["ik"]))
    with pytest.raises(RemoteArmError, match="offset norm"):
        _validate_args(build_parser().parse_args(["ik", "--dx", "0.501"]))


def test_point_defaults_to_bounded_one_shot_vision_approach() -> None:
    args = build_parser().parse_args(["point"])
    _validate_args(args)
    assert args.server == "http://127.0.0.1:8000"
    assert args.standoff == pytest.approx(0.25)
    assert args.max_approach == pytest.approx(0.05)
    assert args.duration == pytest.approx(0.6)
    assert args.no_return is False


def test_fetch_vision_target_requires_fresh_registered_xyz(monkeypatch) -> None:
    payload = {
        "depth_valid": True,
        "target_age_ms": 42.0,
        "track_id": 7,
        "detector_confidence": 0.91,
        "object_xyz_m": [0.55, -0.1, 0.02],
        "predicted_xyz_m": [0.57, -0.1, 0.02],
        "prediction_source": "learned_trajectory",
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        "object_tracking.manual_arm_cli.urllib.request.urlopen",
        lambda url, timeout: Response(),
    )
    target = _fetch_vision_target("http://127.0.0.1:8000/")
    assert target["object_xyz_m"] == pytest.approx((0.55, -0.1, 0.02))
    assert target["predicted_xyz_m"] == pytest.approx((0.57, -0.1, 0.02))
    assert target["target_age_ms"] == pytest.approx(42.0)


def test_fetch_vision_target_rejects_stale_sample(monkeypatch) -> None:
    payload = {
        "depth_valid": True,
        "target_age_ms": 501.0,
        "object_xyz_m": [0.55, -0.1, 0.02],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        "object_tracking.manual_arm_cli.urllib.request.urlopen",
        lambda url, timeout: Response(),
    )
    with pytest.raises(RemoteArmError, match="stale"):
        _fetch_vision_target("http://127.0.0.1:8000")


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


def test_guarded_joint_path_preserves_waypoint_route() -> None:
    path = _guarded_joint_path(
        [
            (0.0, 0.0),
            (0.08, -0.02),
            (0.03, 0.04),
        ]
    )
    assert path[1] == pytest.approx((0.08, -0.02))
    assert path[-1] == pytest.approx((0.03, 0.04))
    complete = [(0.0, 0.0), *path]
    assert (
        max(
            max(abs(end - start) for start, end in zip(before, after))
            for before, after in zip(complete, complete[1:])
        )
        <= 0.05
    )


def test_signed_progress_rejects_motion_opposite_the_requested_direction() -> None:
    assert _signed_progress(0.1, 0.14, 0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.06, -0.05) == pytest.approx(0.04)
    assert _signed_progress(0.1, 0.14, -0.05) == pytest.approx(-0.04)


def test_motion_trace_records_failed_trials(tmp_path) -> None:
    path = _write_motion_trace(
        tmp_path / "failed.json",
        {"ok": False, "error": "following_error"},
        [{"phase": "outward_1", "measured_arm_q": [0.0] * 14}],
    )
    payload = path.read_text(encoding="utf-8")
    assert '"ok": false' in payload
    assert '"outward_1"' in payload
