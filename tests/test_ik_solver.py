from pathlib import Path

from object_tracking.arm_tracking.ik_solver import (
    G1RightArmIK,
    RIGHT_ARM_JOINTS,
    XR_TELEOPERATE_REVISION,
    collision_aware_joint_path,
    default_urdf_path,
)
import numpy as np
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


def test_collision_aware_path_detours_around_blocked_direct_edge() -> None:
    def valid(q: np.ndarray) -> bool:
        return not (0.35 < q[0] < 0.65 and abs(q[1]) < 0.22)

    path = collision_aware_joint_path(
        (0.0, 0.0),
        (1.0, 0.0),
        (-0.2, -1.0),
        (1.2, 1.0),
        valid,
        edge_step_rad=0.02,
        extension_step_rad=0.15,
        seed=7,
    )

    assert path is not None
    assert path[0] == pytest.approx((0.0, 0.0))
    assert path[-1] == pytest.approx((1.0, 0.0))
    assert any(abs(q[1]) > 0.22 for q in path[1:-1])
    for begin, end in zip(path, path[1:]):
        samples = max(1, int(np.ceil(np.max(np.abs(np.subtract(end, begin))) / 0.02)))
        assert all(
            valid((1.0 - alpha) * np.asarray(begin) + alpha * np.asarray(end))
            for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]
        )


def test_collision_aware_path_rejects_invalid_goal() -> None:
    path = collision_aware_joint_path(
        (0.0, 0.0),
        (0.5, 0.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        lambda q: not (0.4 < q[0] < 0.6),
    )
    assert path is None


def test_g1_rest_pose_can_detour_around_right_hip_for_pointing() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    urdf = default_urdf_path(repo_root)
    if not urdf.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    solver = G1RightArmIK(
        urdf,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.5,
        discontinuity_limit_rad=0.35,
        translation_weight=400.0,
        orientation_weight=0.03,
    )
    start_q = np.asarray(
        (0.296993, -0.203672, 0.043167, 0.984252, -0.107451, 0.019762, 0.000875)
    )
    target = solver.forward_kinematics(start_q)
    target[:3, 3] += np.asarray((0.07615, 0.03750, 0.05288))

    route = solver.solve_with_collision_detour(
        target,
        start_q,
        enforce_orientation=False,
    )

    assert route.ok, route.reason
    assert route.q_path is not None
    assert len(route.q_path) >= 3
    assert route.position_error_m < 0.005
