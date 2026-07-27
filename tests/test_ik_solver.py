from pathlib import Path

from object_tracking.arm_tracking.ik_solver import (
    G1RightArmIK,
    RIGHT_ARM_JOINTS,
    XR_TELEOPERATE_REVISION,
    adaptive_table_route,
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


def test_translation_jacobian_rejects_unknown_backend() -> None:
    solver = object.__new__(G1RightArmIK)
    with pytest.raises(ValueError, match="jacobian_backend"):
        solver._translation_jacobian(
            np.zeros(7),
            backend="unknown",
            current_position=np.zeros(3),
        )


def test_zero_motion_local_ik_still_validates_current_collision_edge() -> None:
    solver = object.__new__(G1RightArmIK)
    solver.forward_kinematics = lambda q: np.eye(4)  # type: ignore[method-assign]
    solver.validate_joint_path = (  # type: ignore[method-assign]
        lambda knots, **kwargs: "link_support_region_clearance"
    )

    result = solver.solve_local_translation(
        np.eye(4),
        np.zeros(7),
        support_plane=object(),
    )

    assert not result.ok
    assert result.q_rad is None
    assert result.reason == "link_support_region_clearance"


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


def test_adaptive_table_route_exits_under_table_through_certified_edge() -> None:
    support = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (0.4, -0.3),
        (0.8, 0.3),
        certified_edges=("u_min",),
    )

    route = adaptive_table_route(
        (0.55, 0.0, 0.01),
        (0.65, 0.0, 0.06),
        support,
        maximum_segment_m=0.02,
    )

    assert route.topology == "under_or_inside"
    assert route.exit_edge == "u_min"
    assert len(route.points_xyz_m) > 10
    # It moves behind the inflated near edge before rising.
    assert min(point[0] for point in route.points_xyz_m) < 0.30
    outside_index = next(
        index for index, point in enumerate(route.points_xyz_m) if point[0] < 0.30
    )
    assert max(point[2] for point in route.points_xyz_m[: outside_index + 1]) < 0.10


def test_adaptive_table_route_rises_before_crossing_from_outside() -> None:
    support = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (0.4, -0.3),
        (0.8, 0.3),
        certified_edges=("u_min",),
    )

    route = adaptive_table_route(
        (0.20, 0.0, 0.01),
        (0.65, 0.0, 0.06),
        support,
        maximum_segment_m=0.02,
    )

    assert route.topology == "outside"
    first_over_table = next(
        point for point in route.points_xyz_m if point[0] >= 0.4
    )
    assert first_over_table[2] >= 0.099


def test_adaptive_table_route_refuses_unknown_below_table_exit() -> None:
    support = SupportRegion(
        plane=Plane((0.0, 0.0, 1.0), 0.0),
        origin=np.zeros(3),
        axis_u=np.asarray((1.0, 0.0, 0.0)),
        axis_v=np.asarray((0.0, 1.0, 0.0)),
        minimum_uv=np.asarray((0.4, -0.3)),
        maximum_uv=np.asarray((0.8, 0.3)),
    )

    with pytest.raises(ValueError, match="no certified table edge"):
        adaptive_table_route((0.55, 0.0, 0.01), (0.65, 0.0, 0.06), support)


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


def test_analytic_translation_jacobian_matches_finite_difference() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    urdf = default_urdf_path(repo_root)
    if not urdf.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    solver = G1RightArmIK(urdf)
    q = np.asarray(
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
    current = solver.forward_kinematics(q)[:3, 3]

    analytic = solver._translation_jacobian(
        q,
        backend="analytic",
        current_position=current,
    )
    finite_difference = solver._translation_jacobian(
        q,
        backend="finite_difference",
        current_position=current,
    )

    assert analytic == pytest.approx(finite_difference, abs=3e-5)


def test_global_solver_rejects_unreachable_target() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    urdf = default_urdf_path(repo_root)
    if not urdf.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    solver = G1RightArmIK(urdf)
    seed_q = np.zeros(7)
    target = solver.forward_kinematics(seed_q)
    target[:3, 3] = np.asarray((5.0, -5.0, 5.0))

    result = solver.solve(target, seed_q)

    assert not result.ok
    assert result.q_rad is None
    assert result.reason in {"discontinuous", "position_error"} or (
        result.reason is not None and result.reason.startswith("discontinuous:")
    )


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


def test_lab_g1_hip_rest_pose_has_bounded_guided_table_clearance() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    urdf = default_urdf_path(repo_root)
    if not urdf.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    solver = G1RightArmIK(urdf)
    # Read-only lowstate captured from the lab G1 on 2026-07-27. Its factory
    # rest pose has a 5.9 mm modeled hand/hip overlap, so it must follow the
    # same exit-only path and become strictly collision-free before task IK.
    start_q = (
        0.1428160071,
        -0.0608439110,
        0.0212360471,
        1.4058095217,
        -0.1720934808,
        0.1501144022,
        -0.1068034172,
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

    route = solver.plan_start_collision_escape(start_q, support_plane=support)

    assert route.ok, route.reason
    assert route.q_path is not None
    assert len(route.q_path) > 2
    assert max(
        max(abs(actual - previous) for actual, previous in zip(end, begin))
        for begin, end in zip(route.q_path, route.q_path[1:])
    ) <= 0.04 + 1e-9
    assert solver.validate_joint_path(route.q_path, support_plane=support) is None
