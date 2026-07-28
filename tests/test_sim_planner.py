from __future__ import annotations

import numpy as np

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.ik_solver import IKResult
from object_tracking.arm_tracking.interception import LiveInterceptConfig
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.sim_closed_loop import ObjectObservation, SimState
from object_tracking.arm_tracking.sim_planner import ClosedLoopInterceptionPlanner
from object_tracking.intercept_planner import InterceptConfig


class FakeSolver:
    def forward_kinematics(self, _q):
        transform = np.eye(4)
        transform[:3, 3] = (0.40, -0.08, 0.25)
        return transform

    def collision_labels(self, _q):
        return ()

    def solve_local_translation(self, _target, _q, *, support_plane):
        assert support_plane.source == "test"
        return IKResult(True, (0.01,) * 7, 0.001, 0.0)

    def gravity_compensation_torque(self, _q):
        return (0.2,) * 7


def profile() -> LiveInterceptConfig:
    return LiveInterceptConfig(
        calibration_id="sim",
        plane_point_m=(0.4, 0.0, 0.16),
        plane_normal=(0.0, 1.0, 0.0),
        crossing_minimum_m=(0.3, -0.02, 0.10),
        crossing_maximum_m=(0.5, 0.02, 0.22),
        wanted_classes=("bunny",),
        minimum_confidence=0.5,
        ready_right_arm_q_rad=(0.0,) * 7,
        ready_pose_tolerance_rad=0.1,
        lane_facing_palm_normal=(0.0, 1.0, 0.0),
        minimum_observations=2,
        maximum_residual_m=0.02,
        commit_horizon_s=0.5,
        post_crossing_hold_s=0.1,
        revalidation_tolerance_m=0.03,
        minimum_deadline_slack_s=0.01,
        perception_ttl_s=0.25,
        planner=InterceptConfig(
            compute_delay_s=0.01,
            command_delay_s=0.01,
            maximum_palm_speed_m_s=1.0,
            maximum_palm_acceleration_m_s2=5.0,
            settle_time_s=0.01,
        ),
    )


def state(sequence: int, time_s: float, bunny_y: float = 0.12) -> SimState:
    support = SupportRegion(
        plane=Plane((0.0, 0.0, 1.0), -0.1),
        origin=np.asarray((0.25, 0.3, 0.1)),
        axis_u=np.asarray((1.0, 0.0, 0.0)),
        axis_v=np.asarray((0.0, 1.0, 0.0)),
        minimum_uv=(0.0, -0.6),
        maximum_uv=(0.4, 0.0),
        source="test",
    )
    return SimState(
        episode_id="episode-a",
        sequence=sequence,
        simulation_time_s=time_s,
        calibration_id="sim",
        joint_contract_id=joint_contract_id(),
        body_q_rad=(0.0,) * 29,
        body_dq_rad_s=(0.0,) * 29,
        support_region=support,
        object_observation=ObjectObservation(
            track_id=1,
            class_name="bunny",
            confidence=1.0,
            position_m=(0.4, bunny_y, 0.16),
            velocity_m_s=(0.0, -0.3, 0.0),
            observation_time_s=time_s,
            consecutive_observations=3,
            residual_m=0.001,
        ),
    )


def test_planner_returns_gravity_supported_target_with_ruckig_evidence() -> None:
    planner = ClosedLoopInterceptionPlanner(profile(), FakeSolver())  # type: ignore[arg-type]

    command = planner.plan(state(1, 1.0))

    assert command.status == "target"
    assert command.reason == "intercept_local_translation"
    assert command.right_arm_q_rad == (0.01,) * 7
    assert command.right_arm_tau_ff_nm == (0.2,) * 7
    assert 0.0 < command.remaining_ruckig_duration_s < command.crossing_time_from_now_s
    assert command.target_palm_position_m[1] < 0.0


def test_planner_discards_state_rewind_without_running_ik() -> None:
    planner = ClosedLoopInterceptionPlanner(profile(), FakeSolver())  # type: ignore[arg-type]
    assert planner.plan(state(2, 1.0)).status == "target"

    rejected = planner.plan(state(1, 0.9))

    assert rejected.status == "rejected"
    assert rejected.reason == "stale_or_out_of_order_state"


def test_preview_does_not_emit_motion_before_production_commit_gate() -> None:
    planner = ClosedLoopInterceptionPlanner(profile(), FakeSolver())  # type: ignore[arg-type]

    command = planner.plan(state(1, 1.0, bunny_y=0.6))

    assert command.status == "preview"
    assert command.reason == "waiting_for_commit_window"
    assert command.right_arm_q_rad is None
    assert command.crossing_time_from_now_s == 2.0
