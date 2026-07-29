"""Production-class closed-loop interception planner for simulator state."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .ik_solver import G1RightArmIK, IKResult
from .interception import (
    InterceptObservation,
    InterceptState,
    LiveInterceptConfig,
    LiveInterceptController,
)
from .joints import joint_contract_id
from .runtime import (
    compress_validated_joint_path,
    ruckig_edge_is_valid,
    select_start_escape_waypoint,
)
from .sim_closed_loop import SequenceGate, SimCommand, SimState
from .trajectory import minimum_ruckig_path_duration_s


@dataclass(frozen=True)
class SimPlannerConfig:
    maximum_velocity_rad_s: float = 1.0
    maximum_acceleration_rad_s2: float = 4.0
    maximum_jerk_rad_s3: float = 30.0
    top_clearance_m: float = 0.11
    minimum_link_support_clearance_m: float = 0.055
    path_compression_span_rad: float = 0.70
    path_compression_skip_knots: int = 32


class ClosedLoopInterceptionPlanner:
    """Plan from each newest measured state using production IK and geometry."""

    def __init__(
        self,
        profile: LiveInterceptConfig,
        solver: G1RightArmIK,
        config: SimPlannerConfig | None = None,
    ) -> None:
        self.profile = profile
        self.solver = solver
        self.config = config or SimPlannerConfig()
        self.intercept = LiveInterceptController(profile)
        self.sequence_gate = SequenceGate()
        self._path: tuple[tuple[float, ...], ...] | None = None
        self._path_index = 1
        self._validated_path_index: int | None = None
        self._path_last_advance_q: tuple[float, ...] | None = None
        self._approach_target_position: np.ndarray | None = None
        self._last_observation_time_s: float | None = None

    def plan(self, state: SimState) -> SimCommand:
        started = time.perf_counter()
        try:
            accepted = self.sequence_gate.accept(state)
        except ValueError as exc:
            return self._response(state, "rejected", str(exc), started)
        if not accepted:
            return self._response(state, "rejected", "stale_or_out_of_order_state", started)
        if state.calibration_id != self.profile.calibration_id:
            return self._response(state, "rejected", "calibration_mismatch", started)
        if state.joint_contract_id != joint_contract_id():
            return self._response(state, "rejected", "joint_contract_mismatch", started)

        measured_q = state.right_arm_q_rad
        measured_dq = state.right_arm_dq_rad_s
        palm_position = self.solver.forward_kinematics(measured_q)[:3, 3]
        observation = state.object_observation
        if observation is None:
            decision = self.intercept.current_without_observation(
                now_s=state.simulation_time_s
            )
        else:
            self._last_observation_time_s = observation.observation_time_s
            decision = self.intercept.update(
                InterceptObservation(
                    track_id=observation.track_id,
                    class_name=observation.class_name,
                    confidence=observation.confidence,
                    position_m=observation.position_m,
                    velocity_m_s=observation.velocity_m_s,
                    timestamp_s=observation.observation_time_s,
                    consecutive_observations=observation.consecutive_observations,
                    residual_m=observation.residual_m,
                ),
                now_s=state.simulation_time_s,
                palm_position_m=palm_position,
            )
        if decision.state in (InterceptState.HOLD, InterceptState.EXPIRED):
            if self._path is not None and self._approach_target_position is not None:
                return self._continue_staging_path(
                    state,
                    measured_q,
                    measured_dq,
                    decision.reason,
                    started,
                    decision,
                )
            return self._measured_hold(state, measured_q, decision.reason, started, decision)
        publish_allowed = decision.may_publish or (
            self.profile.preview_staging_enabled and decision.may_stage
        )
        if decision.target_palm_position_m is None or not publish_allowed:
            return self._response(
                state,
                "preview",
                decision.reason,
                started,
                decision=decision,
            )

        target_position = np.asarray(decision.target_palm_position_m, dtype=float)
        target_position = state.support_region.project_to_clearance(
            target_position,
            minimum_clearance_m=0.05,
        )
        next_q, route_reason, remaining_targets = self._next_joint_target(
            measured_q,
            target_position,
            state,
        )
        if next_q is None:
            self.intercept.invalidate(route_reason, now_s=state.simulation_time_s)
            return self._measured_hold(state, measured_q, route_reason, started, decision)

        duration = minimum_ruckig_path_duration_s(
            current_position=measured_q,
            current_velocity=measured_dq,
            target_positions=remaining_targets,
            maximum_velocity=self.config.maximum_velocity_rad_s,
            maximum_acceleration=self.config.maximum_acceleration_rad_s2,
            maximum_jerk=self.config.maximum_jerk_rad_s3,
        )
        crossing_time = decision.crossing_time_from_now_s
        if (
            crossing_time is not None
            and duration + self.profile.minimum_deadline_slack_s > crossing_time
        ):
            self.intercept.invalidate(
                "ruckig_deadline_unreachable",
                now_s=state.simulation_time_s,
            )
            return self._measured_hold(
                state,
                measured_q,
                "ruckig_deadline_unreachable",
                started,
                decision,
            )
        try:
            torque = self.solver.gravity_compensation_torque(next_q)
        except (RuntimeError, ValueError):
            return self._response(
                state,
                "hold",
                "gravity_compensation_failed",
                started,
                decision=decision,
                duration_s=duration,
            )
        return self._response(
            state,
            "target",
            (
                route_reason
                if decision.may_publish
                else f"preview_stage:{route_reason}"
            ),
            started,
            decision=decision,
            duration_s=duration,
            q_rad=next_q,
            tau_nm=torque,
        )

    def _continue_staging_path(
        self,
        state: SimState,
        measured_q: tuple[float, ...],
        measured_dq: tuple[float, ...],
        decision_reason: str,
        started: float,
        decision: Any,
    ) -> SimCommand:
        """Finish an active collision-checked table approach during prediction holds."""

        assert self._approach_target_position is not None
        next_q, route_reason, remaining_targets = self._next_joint_target(
            measured_q,
            self._approach_target_position,
            state,
            path_only=True,
        )
        if next_q is None:
            if route_reason != "approach_complete":
                self.intercept.invalidate(route_reason, now_s=state.simulation_time_s)
            return self._measured_hold(
                state,
                measured_q,
                (
                    decision_reason
                    if route_reason == "approach_complete"
                    else route_reason
                ),
                started,
                decision,
            )
        duration = minimum_ruckig_path_duration_s(
            current_position=measured_q,
            current_velocity=measured_dq,
            target_positions=remaining_targets,
            maximum_velocity=self.config.maximum_velocity_rad_s,
            maximum_acceleration=self.config.maximum_acceleration_rad_s2,
            maximum_jerk=self.config.maximum_jerk_rad_s3,
        )
        try:
            torque = self.solver.gravity_compensation_torque(next_q)
        except (RuntimeError, ValueError):
            return self._response(
                state,
                "hold",
                "gravity_compensation_failed",
                started,
                decision=decision,
                duration_s=duration,
            )
        return self._response(
            state,
            "target",
            f"preview_stage:{route_reason}:prediction_hold:{decision_reason}",
            started,
            decision=decision,
            duration_s=duration,
            q_rad=next_q,
            tau_nm=torque,
        )

    def _measured_hold(
        self,
        state: SimState,
        measured_q: tuple[float, ...],
        reason: str,
        started: float,
        decision: Any,
    ) -> SimCommand:
        """Refresh a stationary gravity-supported target after planning stops."""

        validation_error = self.solver.validate_joint_path(
            (measured_q, measured_q),
            support_plane=state.support_region,
            minimum_support_clearance_m=0.05,
            edge_step_rad=0.010,
            semantic_edge_step_rad=0.005,
            require_escape_cleared=False,
        )
        if validation_error is not None:
            return self._response(
                state,
                "hold",
                f"measured_hold_{validation_error}",
                started,
                decision=decision,
            )
        try:
            torque = self.solver.gravity_compensation_torque(measured_q)
        except (RuntimeError, ValueError):
            return self._response(
                state,
                "hold",
                "gravity_compensation_failed",
                started,
                decision=decision,
            )
        return self._response(
            state,
            "target",
            f"measured_hold:{reason}",
            started,
            decision=decision,
            duration_s=0.0,
            q_rad=measured_q,
            tau_nm=torque,
        )

    def _next_joint_target(
        self,
        measured_q: tuple[float, ...],
        target_position: np.ndarray,
        state: SimState,
        *,
        path_only: bool = False,
    ) -> tuple[
        tuple[float, ...] | None,
        str,
        tuple[tuple[float, ...], ...],
    ]:
        current_transform = self.solver.forward_kinematics(measured_q)
        start_topology = state.support_region.classify_point(
            current_transform[:3, 3],
            side_margin_m=0.10,
            top_clearance_m=0.10,
        )
        needs_approach = (
            self._path is not None
            or bool(self.solver.collision_labels(measured_q))
            or start_topology != "above_clearance"
        )
        if needs_approach:
            if self._path is None:
                target_transform = current_transform.copy()
                target_transform[:3, 3] = target_position
                approach = self.solver.plan_adaptive_table_approach(
                    target_transform,
                    measured_q,
                    support_plane=state.support_region,
                    top_clearance_m=self.config.top_clearance_m,
                    final_validation_edge_step_rad=None,
                    desired_palm_normal=self.profile.lane_facing_palm_normal,
                    maximum_palm_normal_error_rad=(
                        self.profile.planner.maximum_orientation_error_rad
                    ),
                )
                route = "adaptive_table_approach"
                if not approach.ok or approach.q_path is None:
                    approach = self.solver.plan_guided_clearance(
                        measured_q,
                        support_plane=state.support_region,
                        lift_m=0.12,
                        forward_m=0.04,
                    )
                    route = "guided_table_clearance_fallback"
                if not approach.ok or approach.q_path is None:
                    return None, f"ik_{approach.reason or 'approach_failed'}", ()
                self._path = compress_validated_joint_path(
                    approach.q_path,
                    lambda begin, end: ruckig_edge_is_valid(
                        self.solver,
                        begin,
                        end,
                        support_plane=state.support_region,
                        maximum_velocity_rad_s=self.config.maximum_velocity_rad_s,
                        maximum_acceleration_rad_s2=(
                            self.config.maximum_acceleration_rad_s2
                        ),
                        maximum_jerk_rad_s3=self.config.maximum_jerk_rad_s3,
                        minimum_support_clearance_m=(
                            self.config.minimum_link_support_clearance_m
                        ),
                    ),
                    maximum_span_rad=self.config.path_compression_span_rad,
                    maximum_skip_knots=self.config.path_compression_skip_knots,
                )
                if self._path is None:
                    return None, "ik_approach_path_compression", ()
                self._path_index = 1
                self._approach_target_position = target_position.copy()
                self._path_last_advance_q = measured_q
                # Every compressed edge has already passed the complete chord
                # and synchronized Ruckig collision/support validator.
                self._validated_path_index = 1
            previous_path_index = self._path_index
            waypoint, self._path_index, error = select_start_escape_waypoint(
                measured_q,
                self._path,
                self._path_index,
                reached_tolerance_rad=0.018,
                last_advance_q_rad=self._path_last_advance_q,
                residual_lookahead_rad=0.035,
                measured_velocity_rad_s=state.right_arm_dq_rad_s,
                # Match the live runtime: hand off after measured waypoint
                # arrival while preserving the prevalidated edge sequence.
                maximum_waypoint_velocity_rad_s=0.10,
                final_reached_tolerance_rad=0.025,
                final_maximum_waypoint_velocity_rad_s=0.10,
            )
            if self._path_index > previous_path_index:
                # Advancement requires measured arrival within 1 mrad with
                # velocity below 0.005 rad/s. Reuse the prevalidated next edge
                # only after this near-zero-state handoff.
                self._validated_path_index = self._path_index
                self._path_last_advance_q = measured_q
            if error is not None:
                self._reset_path()
                return None, f"ik_{error}", ()
            if waypoint is not None:
                # The bridge keeps one online Ruckig trajectory alive while a
                # waypoint is active. Recalculating an offline zero-acceleration
                # trajectory from every mid-edge measurement can invent a
                # different overshoot and reject an edge that is already being
                # executed safely. Validate once when the waypoint changes;
                # measured-state corridor checks above remain active per frame.
                if self._validated_path_index != self._path_index:
                    if not ruckig_edge_is_valid(
                        self.solver,
                        measured_q,
                        waypoint,
                        support_plane=state.support_region,
                        maximum_velocity_rad_s=self.config.maximum_velocity_rad_s,
                        maximum_acceleration_rad_s2=self.config.maximum_acceleration_rad_s2,
                        maximum_jerk_rad_s3=self.config.maximum_jerk_rad_s3,
                        minimum_support_clearance_m=(
                            self.config.minimum_link_support_clearance_m
                        ),
                        current_velocity_rad_s=state.right_arm_dq_rad_s,
                    ):
                        self._reset_path()
                        return None, "ik_measured_edge:ruckig_path_invalid", ()
                    self._validated_path_index = self._path_index
                return (
                    waypoint,
                    route if "route" in locals() else "table_approach",
                    self._path[self._path_index :],
                )
            self._reset_path()
            if path_only:
                return None, "approach_complete", ()

        target_transform = self.solver.forward_kinematics(measured_q)
        target_transform[:3, 3] = target_position
        local: IKResult = self.solver.solve_local_translation(
            target_transform,
            measured_q,
            support_plane=state.support_region,
            minimum_support_clearance_m=(
                self.config.minimum_link_support_clearance_m
            ),
        )
        if not local.ok or local.q_rad is None:
            return None, f"ik_{local.reason or 'local_translation_failed'}", ()
        return local.q_rad, "intercept_local_translation", (local.q_rad,)

    def _reset_path(self) -> None:
        self._path = None
        self._path_index = 1
        self._validated_path_index = None
        self._path_last_advance_q = None
        self._approach_target_position = None

    def _response(
        self,
        state: SimState,
        status: str,
        reason: str,
        started: float,
        *,
        decision: Any | None = None,
        duration_s: float | None = None,
        q_rad: tuple[float, ...] | None = None,
        tau_nm: tuple[float, ...] | None = None,
    ) -> SimCommand:
        return SimCommand(
            episode_id=state.episode_id,
            state_sequence=state.sequence,
            simulation_time_s=state.simulation_time_s,
            source_observation_time_s=(
                self._last_observation_time_s
            ),
            status=status,
            reason=reason,
            right_arm_q_rad=q_rad,
            right_arm_tau_ff_nm=tau_nm,
            target_palm_position_m=(
                None if decision is None else decision.target_palm_position_m
            ),
            predicted_crossing_m=(
                None if decision is None else decision.predicted_crossing_m
            ),
            crossing_time_from_now_s=(
                None if decision is None else decision.crossing_time_from_now_s
            ),
            remaining_ruckig_duration_s=duration_s,
            arrival_slack_s=(
                None if decision is None else decision.arrival_slack_s
            ),
            planning_latency_ms=(time.perf_counter() - started) * 1000.0,
            ik_step_type=reason if q_rad is not None else None,
        )
