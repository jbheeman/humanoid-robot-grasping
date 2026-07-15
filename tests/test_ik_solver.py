from pathlib import Path

from object_tracking.arm_tracking.ik_solver import (
    G1RightArmIK,
    RIGHT_ARM_JOINTS,
    XR_TELEOPERATE_REVISION,
    default_urdf_path,
)
import pytest


def test_pinned_revision_and_joint_order_are_explicit() -> None:
    assert len(XR_TELEOPERATE_REVISION) == 40
    assert RIGHT_ARM_JOINTS[0] == "right_shoulder_pitch_joint"
    assert RIGHT_ARM_JOINTS[-1] == "right_wrist_yaw_joint"
    assert len(RIGHT_ARM_JOINTS) == 7


def test_default_urdf_uses_ignored_dependency_tree() -> None:
    path = default_urdf_path(Path("/repo"))
    assert path == Path("/repo/.deps/xr_teleoperate/assets/g1/g1_body29_hand14.urdf")


def test_forward_kinematics_rejects_invalid_joint_shape() -> None:
    solver = object.__new__(G1RightArmIK)
    with pytest.raises(ValueError, match="seven finite"):
        solver.forward_kinematics([0.0] * 6)
