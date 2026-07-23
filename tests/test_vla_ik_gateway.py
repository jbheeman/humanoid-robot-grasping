from types import SimpleNamespace

import numpy as np

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.unifolm_vla import compose_pose23, parse_action_chunk
from object_tracking.vla_ik_gateway import GeometricIKGateway, IKGatewayConfig


def support() -> SupportRegion:
    return SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        certified_edges=("u_min", "u_max", "v_min", "v_max"),
    )


def waypoint(position: tuple[float, float, float]) -> object:
    right = np.eye(4)
    right[:3, 3] = position
    values = compose_pose23(
        np.eye(4),
        right,
        right_gripper=4.0,
        left_gripper=4.0,
        waist_yaw_roll_pitch=(0.0, 0.0, 0.0),
    )
    return parse_action_chunk([values])[0]


def test_gateway_projects_then_runs_identical_swept_geometry_checks() -> None:
    start = (0.0,) * 7
    moved = (0.01,) * 7

    class Solver:
        def solve_local_translation(
            self,
            target: np.ndarray,
            seed: object,
            **kwargs: object,
        ) -> object:
            np.testing.assert_allclose(target[:3, 3], (0.2, 0.0, 0.05))
            assert tuple(seed) == start
            assert kwargs["support_plane"] is table
            assert kwargs["maximum_joint_step_rad"] == 0.025
            assert kwargs["validate_path"] is True
            return SimpleNamespace(ok=True, q_rad=moved, reason=None)

        def validate_joint_path(self, path: object, *, support_plane: object) -> None:
            assert tuple(path) == (start, moved)
            assert support_plane is table
            return None

    table = support()
    result = GeometricIKGateway(Solver()).plan(
        (waypoint((0.2, 0.0, -0.01)),),
        start,
        table,
    )

    assert result.ok
    assert result.q_path == (moved,)
    assert result.projected_waypoints == 1
    assert np.isclose(result.maximum_projection_m, 0.06)
    assert result.intervention_fraction == 1.0


def test_gateway_rejects_large_projection_before_calling_ik() -> None:
    class Solver:
        def solve_local_translation(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("IK must not receive a grossly unsafe proposal")

    result = GeometricIKGateway(
        Solver(),
        IKGatewayConfig(maximum_table_projection_m=0.08),
    ).plan(
        (waypoint((0.2, 0.0, -0.20)),),
        (0.0,) * 7,
        support(),
    )

    assert not result.ok
    assert result.q_path == ()
    assert result.reason is not None
    assert result.reason.startswith("table_projection_limit:")


def test_gateway_fails_closed_on_final_swept_path_rejection() -> None:
    class Solver:
        def solve_local_translation(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return SimpleNamespace(ok=True, q_rad=(0.01,) * 7, reason=None)

        def validate_joint_path(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            return "table_edge_collision"

    result = GeometricIKGateway(Solver()).plan(
        (waypoint((0.2, 0.0, 0.10)),),
        (0.0,) * 7,
        support(),
    )

    assert not result.ok
    assert result.q_path == ()
    assert result.reason == "swept_path_rejected:table_edge_collision"
