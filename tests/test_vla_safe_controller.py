from types import SimpleNamespace

import numpy as np

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.unifolm_vla import compose_pose23
from object_tracking.vla_chunk_scheduler import (
    SafetySignal,
    ScheduleDecision,
    SchedulerConfig,
)
from object_tracking.vla_ik_gateway import IKGatewayConfig
from object_tracking.vla_safe_controller import (
    SafeControllerConfig,
    SafeVLAControllerCore,
)


def support() -> SupportRegion:
    return SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        certified_edges=("u_min", "u_max", "v_min", "v_max"),
    )


def action(right_xyz: tuple[float, float, float]) -> np.ndarray:
    right = np.eye(4)
    right[:3, 3] = right_xyz
    return np.asarray(
        [
            compose_pose23(
                np.eye(4),
                right,
                right_gripper=4.0,
                left_gripper=4.0,
                waist_yaw_roll_pitch=(0.0, 0.0, 0.0),
            )
        ],
        dtype=float,
    )


class Solver:
    def solve_local_translation(
        self,
        target: np.ndarray,
        seed: object,
        **kwargs: object,
    ) -> object:
        del seed
        assert kwargs["validate_path"] is True
        assert kwargs["support_plane"] is not None
        self.last_target = target.copy()
        return SimpleNamespace(ok=True, q_rad=(0.01,) * 7, reason=None)

    def validate_joint_path(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None


def test_controller_runs_each_async_waypoint_through_live_geometric_ik() -> None:
    solver = Solver()
    controller = SafeVLAControllerCore(
        solver,
        SafeControllerConfig(
            scheduler=SchedulerConfig(maximum_step=1.0),
            gateway=IKGatewayConfig(maximum_waypoints=1),
        ),
    )
    accepted = controller.submit_chunk(
        action((0.2, 0.0, -0.01)),
        observation_time_s=10.0,
        inference_completed_time_s=10.2,
        now_s=10.2,
    )
    assert accepted.decision is ScheduleDecision.EXECUTE

    result = controller.tick(
        now_s=10.21,
        measured_q_rad=(0.0,) * 7,
        support=support(),
        support_age_s=0.02,
        safety=SafetySignal(),
    )

    assert result.decision is ScheduleDecision.EXECUTE
    assert result.q_target_rad == (0.01,) * 7
    assert result.projected
    np.testing.assert_allclose(solver.last_target[:3, 3], (0.2, 0.0, 0.05))


def test_controller_cancels_active_chunk_when_table_geometry_is_stale() -> None:
    controller = SafeVLAControllerCore(Solver())
    controller.submit_chunk(
        action((0.2, 0.0, 0.1)),
        observation_time_s=10.0,
        inference_completed_time_s=10.2,
        now_s=10.2,
    )

    result = controller.tick(
        now_s=10.21,
        measured_q_rad=(0.0,) * 7,
        support=support(),
        support_age_s=0.75,
        safety=SafetySignal(),
    )

    assert result.decision is ScheduleDecision.HOLD
    assert result.q_target_rad is None
    assert result.reason == "support_geometry_stale"


def test_controller_never_calls_ik_after_contact_veto() -> None:
    class ForbiddenSolver:
        def solve_local_translation(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("IK must not run after contact")

    controller = SafeVLAControllerCore(ForbiddenSolver())
    controller.submit_chunk(
        action((0.2, 0.0, 0.1)),
        observation_time_s=10.0,
        inference_completed_time_s=10.2,
        now_s=10.2,
    )
    result = controller.tick(
        now_s=10.21,
        measured_q_rad=(0.0,) * 7,
        support=support(),
        support_age_s=0.01,
        safety=SafetySignal(contact=True),
    )

    assert result.decision is ScheduleDecision.CANCEL
    assert result.reason == "contact_detected"
