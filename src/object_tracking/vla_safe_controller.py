"""Transport-free receding-horizon VLA plus geometric-IK controller core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from object_tracking.arm_tracking.geometry import SupportRegion
from object_tracking.unifolm_vla import parse_action_chunk
from object_tracking.vla_chunk_scheduler import (
    SafetySignal,
    ScheduleDecision,
    ScheduledAction,
    SchedulerConfig,
    VLAChunkScheduler,
)
from object_tracking.vla_ik_gateway import (
    GeometricIKGateway,
    IKGatewayConfig,
    IKGatewayResult,
)


@dataclass(frozen=True)
class SafeControllerConfig:
    scheduler: SchedulerConfig = SchedulerConfig()
    gateway: IKGatewayConfig = IKGatewayConfig(maximum_waypoints=1)
    maximum_support_age_s: float = 0.50

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.maximum_support_age_s)
            or self.maximum_support_age_s <= 0.0
        ):
            raise ValueError("maximum support age must be finite and positive")
        if self.gateway.maximum_waypoints != 1:
            raise ValueError("receding-horizon gateway must validate one waypoint per tick")


@dataclass(frozen=True)
class SafeControlResult:
    decision: ScheduleDecision
    q_target_rad: tuple[float, ...] | None
    reason: str
    observation_age_s: float | None = None
    projected: bool = False
    waypoint_index: int | None = None
    waypoint_time_s: float | None = None
    expired_waypoints: int = 0


class SafeVLAControllerCore:
    """Consume asynchronous VLA chunks and emit one guarded joint target.

    The caller owns image capture, inference threads, and the robot transport.
    This class intentionally owns none of them, which makes it possible to fuzz
    the complete scheduling and geometry contract without connecting to ROS.
    """

    def __init__(
        self,
        solver: Any,
        config: SafeControllerConfig | None = None,
    ) -> None:
        self.config = config or SafeControllerConfig()
        self.scheduler = VLAChunkScheduler(23, self.config.scheduler)
        self.gateway = GeometricIKGateway(solver, self.config.gateway)

    def submit_chunk(
        self,
        actions: np.ndarray,
        *,
        observation_time_s: float,
        inference_completed_time_s: float,
        now_s: float,
        state_anchor_time_s: float | None = None,
        inference_started_time_s: float | None = None,
    ) -> ScheduledAction:
        # Parse before queueing so malformed 6D rotations never become active.
        try:
            parse_action_chunk(actions)
        except (TypeError, ValueError):
            return ScheduledAction(
                ScheduleDecision.REJECT,
                None,
                "invalid_unifolm_chunk",
            )
        return self.scheduler.submit(
            actions,
            observation_time_s=observation_time_s,
            inference_completed_time_s=inference_completed_time_s,
            now_s=now_s,
            state_anchor_time_s=state_anchor_time_s,
            inference_started_time_s=inference_started_time_s,
        )

    def tick(
        self,
        *,
        now_s: float,
        measured_q_rad: tuple[float, ...],
        support: SupportRegion | None,
        support_age_s: float,
        safety: SafetySignal,
    ) -> SafeControlResult:
        if (
            support is None
            or not np.isfinite(support_age_s)
            or support_age_s < 0.0
            or support_age_s > self.config.maximum_support_age_s
        ):
            self.scheduler.cancel("support_geometry_stale")
            return SafeControlResult(
                ScheduleDecision.HOLD,
                None,
                "support_geometry_stale",
            )

        scheduled = self.scheduler.tick(now_s=now_s, safety=safety)
        if scheduled.decision is not ScheduleDecision.EXECUTE:
            return SafeControlResult(
                scheduled.decision,
                None,
                scheduled.reason,
                scheduled.observation_age_s,
                waypoint_index=scheduled.waypoint_index,
                waypoint_time_s=scheduled.waypoint_time_s,
                expired_waypoints=scheduled.expired_waypoints,
            )
        assert scheduled.action is not None
        try:
            waypoint = parse_action_chunk((scheduled.action,))[0]
        except ValueError:
            self.scheduler.cancel("scheduled_waypoint_invalid")
            return SafeControlResult(
                ScheduleDecision.HOLD,
                None,
                "scheduled_waypoint_invalid",
                scheduled.observation_age_s,
                waypoint_index=scheduled.waypoint_index,
                waypoint_time_s=scheduled.waypoint_time_s,
                expired_waypoints=scheduled.expired_waypoints,
            )
        result: IKGatewayResult = self.gateway.plan(
            (waypoint,),
            measured_q_rad,
            support,
        )
        if not result.ok or not result.q_path:
            self.scheduler.cancel("geometric_ik_rejected")
            return SafeControlResult(
                ScheduleDecision.HOLD,
                None,
                f"geometric_ik_rejected:{result.reason}",
                scheduled.observation_age_s,
                waypoint_index=scheduled.waypoint_index,
                waypoint_time_s=scheduled.waypoint_time_s,
                expired_waypoints=scheduled.expired_waypoints,
            )
        return SafeControlResult(
            ScheduleDecision.EXECUTE,
            result.q_path[0],
            "guarded",
            scheduled.observation_age_s,
            projected=result.projected_waypoints > 0,
            waypoint_index=scheduled.waypoint_index,
            waypoint_time_s=scheduled.waypoint_time_s,
            expired_waypoints=scheduled.expired_waypoints,
        )
