from dataclasses import replace

import pytest

from object_tracking.arm_tracking.arm_bridge import RobotState
from object_tracking.arm_tracking.joints import BODY_JOINT_NAMES
from object_tracking.arm_tracking.visualization import (
    compose_commanded_body_pose,
    visualization_state,
)


def test_canonical_body_joint_order_and_legacy_arm_projection() -> None:
    assert len(BODY_JOINT_NAMES) == 29
    assert BODY_JOINT_NAMES[0] == "left_hip_pitch_joint"
    assert BODY_JOINT_NAMES[12:15] == (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    assert BODY_JOINT_NAMES[-1] == "right_wrist_yaw_joint"
    state = RobotState(
        arm_q=tuple(float(index) for index in range(14)),
        arm_dq=tuple(float(index + 20) for index in range(14)),
        waist_q=(1.0, 2.0, 3.0),
        received_at=10.0,
        standing=True,
        standing_since=0.0,
    )
    assert state.body_q[12:15] == state.waist_q
    assert state.body_q[15:] == state.arm_q
    assert state.body_dq[15:] == state.arm_dq
    assert replace(state, received_at=11.0).arm_q == state.arm_q


def test_commanded_body_retains_legs_and_waist_and_replaces_arms() -> None:
    measured = [float(index) for index in range(29)]
    commanded = [100.0 + index for index in range(14)]
    result = compose_commanded_body_pose(measured, commanded)
    assert result is not None
    assert result[:15] == measured[:15]
    assert result[15:] == commanded
    with pytest.raises(ValueError):
        compose_commanded_body_pose(measured, [1.0])


def test_visualization_freshness_unavailable_and_fault_schema() -> None:
    fresh = visualization_state(
        measured_body_q=[0.0] * 29,
        measured_body_dq=[0.1] * 29,
        reference_body_q=[0.2] * 29,
        commanded_arm_q=[0.3] * 14,
        received_at=9.9,
        now=10.0,
        state_ttl_s=0.25,
        selected_joint=BODY_JOINT_NAMES[22],
        faulted_joints=(BODY_JOINT_NAMES[22], "unknown"),
    )
    assert fresh["schema_version"] == 1
    assert fresh["read_only"] is True
    assert fresh["fresh"] is True
    assert fresh["commanded_pose_rad"][:15] == [0.0] * 15
    assert fresh["faulted_joints"] == [BODY_JOINT_NAMES[22]]
    stale = visualization_state(
        measured_body_q=[0.0] * 29, received_at=1.0, now=2.0, state_ttl_s=0.25
    )
    assert stale["available"] is True and stale["fresh"] is False
    unavailable = visualization_state(measured_body_q=None, available=False)
    assert unavailable["available"] is False
    assert unavailable["measured_pose_rad"] is None
