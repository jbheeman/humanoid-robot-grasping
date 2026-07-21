from pathlib import Path

from object_tracking.arm_tracking.ik_solver import (
    G1RightArmIK,
    RIGHT_ARM_JOINTS,
    XR_TELEOPERATE_REVISION,
    collision_aware_joint_path,
    default_urdf_path,
)
from object_tracking.arm_tracking.geometry import Plane, SupportRegion
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


def test_collision_aware_path_advances_to_last_valid_extension_prefix() -> None:
    def valid(q: np.ndarray) -> bool:
        return not (0.30 < q[0] < 0.70 and abs(q[1]) < 0.08)

    path = collision_aware_joint_path(
        (0.0, 0.0),
        (1.0, 0.0),
        (-0.2, -1.0),
        (1.2, 1.0),
        valid,
        edge_step_rad=0.01,
        extension_step_rad=0.12,
        max_iterations=5,
        seed=4,
        guided_sampling=False,
    )

    assert path is not None
    assert len(path) == 3


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
        (
            0.2951954305,
            -0.1279316097,
            0.0029960563,
            0.9830660224,
            -0.1099193171,
            0.0618985221,
            -0.0382895991,
        )
    )
    plane = Plane(
        (0.0612211654, -0.0322558272, 0.9976028922),
        0.0058313740,
    )
    corners = []
    for x, y in ((0.42, 0.30), (0.42, -0.30), (0.75, -0.30), (0.75, 0.30)):
        z = -(plane.offset + plane.normal[0] * x + plane.normal[1] * y) / plane.normal[2]
        corners.append((x, y, z))
    support = SupportRegion.from_ordered_corners(plane, corners)
    target = solver.forward_kinematics(start_q)
    target[:3, 3] += np.asarray((0.06, 0.0, 0.12))

    route = solver.solve_with_collision_detour(
        target,
        start_q,
        enforce_orientation=False,
        support_plane=support,
    )

    assert route.ok, route.reason
    assert route.q_path is not None
    assert len(route.q_path) >= 3
    assert route.position_error_m < 0.005
    assert solver.validate_joint_path(route.q_path, support_plane=support) is None


def test_current_g1_hip_rest_pose_has_bounded_guided_table_clearance() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    urdf = default_urdf_path(repo_root)
    if not urdf.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    solver = G1RightArmIK(urdf)
    # Read-only lowstate captured from the stock hip-rest pose.  It contains a
    # shallow 0.21 mm hand/hip mesh overlap that must use the exit-only path.
    start_q = (
        -0.0963891223,
        0.0183118954,
        0.2851406634,
        1.4834433794,
        -0.0324293151,
        -0.2624904811,
        -0.0487997644,
    )
    plane = Plane((0.0, 0.0, 1.0), -0.05)
    support = SupportRegion.from_ordered_corners(
        plane,
        (
            (0.37, 0.35, 0.05),
            (0.37, -0.35, 0.05),
            (0.80, -0.35, 0.05),
            (0.80, 0.35, 0.05),
        ),
    )

    route = solver.plan_guided_clearance(start_q, support_plane=support)

    assert route.ok, route.reason
    assert route.q_path is not None
    assert len(route.q_path) > 2
    assert max(
        max(abs(actual - previous) for actual, previous in zip(end, begin))
        for begin, end in zip(route.q_path, route.q_path[1:])
    ) <= 0.04 + 1e-9
    assert solver.validate_joint_path(route.q_path, support_plane=support) is None
