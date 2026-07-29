"""Deterministic live interception decisions independent of ROS and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from object_tracking.intercept_planner import (
    InterceptConfig,
    InterceptPlan,
    plan_plane_intercept,
)


_SCHEMA_VERSION = 1


def _finite_vector(value: Iterable[float], length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite values") from exc
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _unit_vector(value: Iterable[float], name: str) -> tuple[float, float, float]:
    result = np.asarray(_finite_vector(value, 3, name), dtype=float)
    norm = float(np.linalg.norm(result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"{name} must be a unit vector")
    return tuple(float(item) for item in result)


@dataclass(frozen=True)
class LiveInterceptConfig:
    calibration_id: str
    plane_point_m: tuple[float, float, float]
    plane_normal: tuple[float, float, float]
    crossing_minimum_m: tuple[float, float, float]
    crossing_maximum_m: tuple[float, float, float]
    wanted_classes: tuple[str, ...]
    minimum_confidence: float
    ready_right_arm_q_rad: tuple[float, ...]
    ready_pose_tolerance_rad: float
    lane_facing_palm_normal: tuple[float, float, float]
    minimum_observations: int
    maximum_residual_m: float
    commit_horizon_s: float
    post_crossing_hold_s: float
    revalidation_tolerance_m: float
    minimum_deadline_slack_s: float
    perception_ttl_s: float
    planner: InterceptConfig
    preview_staging_enabled: bool = False
    validated_for_execution: bool = False
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported intercept schema_version {self.schema_version}")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id is required")
        object.__setattr__(
            self, "plane_point_m", _finite_vector(self.plane_point_m, 3, "plane_point_m")
        )
        object.__setattr__(
            self, "plane_normal", _unit_vector(self.plane_normal, "plane_normal")
        )
        object.__setattr__(
            self,
            "crossing_minimum_m",
            _finite_vector(self.crossing_minimum_m, 3, "crossing_minimum_m"),
        )
        object.__setattr__(
            self,
            "crossing_maximum_m",
            _finite_vector(self.crossing_maximum_m, 3, "crossing_maximum_m"),
        )
        if any(
            lower > upper
            for lower, upper in zip(self.crossing_minimum_m, self.crossing_maximum_m)
        ):
            raise ValueError("crossing bounds are inverted")
        if not all(isinstance(token, str) for token in self.wanted_classes):
            raise ValueError("wanted_classes must contain only strings")
        classes = tuple(token.strip().lower() for token in self.wanted_classes if token.strip())
        if not classes:
            raise ValueError("wanted_classes must not be empty")
        object.__setattr__(self, "wanted_classes", classes)
        if not 0.0 < self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in (0, 1]")
        object.__setattr__(
            self,
            "ready_right_arm_q_rad",
            _finite_vector(self.ready_right_arm_q_rad, 7, "ready_right_arm_q_rad"),
        )
        object.__setattr__(
            self,
            "lane_facing_palm_normal",
            _unit_vector(self.lane_facing_palm_normal, "lane_facing_palm_normal"),
        )
        positive = {
            "ready_pose_tolerance_rad": self.ready_pose_tolerance_rad,
            "maximum_residual_m": self.maximum_residual_m,
            "commit_horizon_s": self.commit_horizon_s,
            "post_crossing_hold_s": self.post_crossing_hold_s,
            "revalidation_tolerance_m": self.revalidation_tolerance_m,
            "minimum_deadline_slack_s": self.minimum_deadline_slack_s,
            "perception_ttl_s": self.perception_ttl_s,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("intercept timing and tolerance values must be finite and positive")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.perception_ttl_s > 0.500:
            raise ValueError("perception_ttl_s cannot exceed the robot bridge 500 ms limit")
        if self.post_crossing_hold_s > self.perception_ttl_s:
            raise ValueError("post_crossing_hold_s cannot exceed perception_ttl_s")
        if self.commit_horizon_s > self.planner.maximum_crossing_horizon_s:
            raise ValueError("commit_horizon_s exceeds the planner crossing horizon")
        if not isinstance(self.preview_staging_enabled, bool):
            raise ValueError("preview_staging_enabled must be a boolean")

    def track_matches(self, class_name: str, confidence: float) -> bool:
        normalized = class_name.strip().lower()
        return confidence >= self.minimum_confidence and any(
            token in normalized for token in self.wanted_classes
        )

    def crossing_inside_bounds(self, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= np.asarray(self.crossing_minimum_m))
            and np.all(point <= np.asarray(self.crossing_maximum_m))
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sequence(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def load_live_intercept_config(path: str | Path) -> LiveInterceptConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    value = _mapping(raw, "intercept configuration")
    planner_value = _mapping(value.get("planner"), "planner")
    return LiveInterceptConfig(
        schema_version=int(value.get("schema_version", 0)),
        calibration_id=str(value.get("calibration_id", "")),
        plane_point_m=_sequence(value.get("plane_point_m"), "plane_point_m"),
        plane_normal=_sequence(value.get("plane_normal"), "plane_normal"),
        crossing_minimum_m=_sequence(
            value.get("crossing_minimum_m"),
            "crossing_minimum_m",
        ),
        crossing_maximum_m=_sequence(
            value.get("crossing_maximum_m"),
            "crossing_maximum_m",
        ),
        wanted_classes=_sequence(value.get("wanted_classes"), "wanted_classes"),
        minimum_confidence=float(value.get("minimum_confidence", 0.0)),
        ready_right_arm_q_rad=_sequence(
            value.get("ready_right_arm_q_rad"),
            "ready_right_arm_q_rad",
        ),
        ready_pose_tolerance_rad=float(value.get("ready_pose_tolerance_rad", 0.0)),
        lane_facing_palm_normal=_sequence(
            value.get("lane_facing_palm_normal"),
            "lane_facing_palm_normal",
        ),
        minimum_observations=int(value.get("minimum_observations", 0)),
        maximum_residual_m=float(value.get("maximum_residual_m", 0.0)),
        commit_horizon_s=float(value.get("commit_horizon_s", 0.0)),
        post_crossing_hold_s=float(value.get("post_crossing_hold_s", 0.0)),
        revalidation_tolerance_m=float(value.get("revalidation_tolerance_m", 0.0)),
        minimum_deadline_slack_s=float(value.get("minimum_deadline_slack_s", 0.0)),
        perception_ttl_s=float(value.get("perception_ttl_s", 0.0)),
        validated_for_execution=_boolean(
            value.get("validated_for_execution", False),
            "validated_for_execution",
        ),
        preview_staging_enabled=_boolean(
            value.get("preview_staging_enabled", False),
            "preview_staging_enabled",
        ),
        planner=InterceptConfig(
            compute_delay_s=float(planner_value.get("compute_delay_s", 0.0)),
            command_delay_s=float(planner_value.get("command_delay_s", 0.0)),
            maximum_palm_speed_m_s=float(
                planner_value.get("maximum_palm_speed_m_s", 0.0)
            ),
            maximum_palm_acceleration_m_s2=float(
                planner_value.get("maximum_palm_acceleration_m_s2", 0.0)
            ),
            settle_time_s=float(planner_value.get("settle_time_s", 0.0)),
            bunny_contact_radius_m=float(
                planner_value.get("bunny_contact_radius_m", 0.0)
            ),
            palm_half_thickness_m=float(
                planner_value.get("palm_half_thickness_m", 0.0)
            ),
            minimum_object_speed_m_s=float(
                planner_value.get("minimum_object_speed_m_s", 0.0)
            ),
            maximum_crossing_horizon_s=float(
                planner_value.get("maximum_crossing_horizon_s", 0.0)
            ),
            maximum_orientation_error_rad=float(
                planner_value.get("maximum_orientation_error_rad", 0.0)
            ),
        ),
    )


class InterceptState(str, Enum):
    ACQUIRING = "ACQUIRING"
    PREVIEW = "PREVIEW"
    COMMITTED = "COMMITTED"
    EXPIRED = "EXPIRED"
    HOLD = "HOLD"


@dataclass(frozen=True)
class InterceptObservation:
    track_id: int
    class_name: str
    confidence: float
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    timestamp_s: float
    consecutive_observations: int
    residual_m: float

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        object.__setattr__(
            self, "position_m", _finite_vector(self.position_m, 3, "position_m")
        )
        object.__setattr__(
            self, "velocity_m_s", _finite_vector(self.velocity_m_s, 3, "velocity_m_s")
        )
        if (
            not math.isfinite(self.timestamp_s)
            or self.timestamp_s < 0.0
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
            or not math.isfinite(self.residual_m)
            or self.residual_m < 0.0
            or not isinstance(self.consecutive_observations, int)
            or self.consecutive_observations < 1
        ):
            raise ValueError("observation timing, confidence, and residual are invalid")


@dataclass(frozen=True)
class InterceptDecision:
    state: InterceptState
    reason: str
    target_palm_position_m: tuple[float, float, float] | None
    predicted_crossing_m: tuple[float, float, float] | None
    crossing_time_from_now_s: float | None
    arrival_slack_s: float | None
    last_confirmation_age_s: float | None
    plan: InterceptPlan | None

    @property
    def may_publish(self) -> bool:
        return self.state is InterceptState.COMMITTED

    @property
    def may_stage(self) -> bool:
        return (
            self.state is InterceptState.PREVIEW
            and self.target_palm_position_m is not None
        )


class LiveInterceptController:
    """Preview, latch, reconfirm, and expire one bounded plane interception."""

    def __init__(self, config: LiveInterceptConfig) -> None:
        self.config = config
        self.state = InterceptState.ACQUIRING
        self.active_track_id: int | None = None
        self.last_observation_timestamp_s: float | None = None
        self.latched_target: np.ndarray | None = None
        self.crossing_deadline_s: float | None = None
        self.last_confirmation_s: float | None = None
        self.terminal_reason: str | None = None

    def reset(self) -> None:
        self.state = InterceptState.ACQUIRING
        self.active_track_id = None
        self.last_observation_timestamp_s = None
        self.latched_target = None
        self.crossing_deadline_s = None
        self.last_confirmation_s = None
        self.terminal_reason = None

    def invalidate(self, reason: str, *, now_s: float) -> InterceptDecision:
        if not reason:
            raise ValueError("invalidation reason is required")
        return self._fail(reason, now_s)

    def update(
        self,
        observation: InterceptObservation,
        *,
        now_s: float,
        palm_position_m: Iterable[float],
    ) -> InterceptDecision:
        if not math.isfinite(now_s) or now_s < observation.timestamp_s:
            return self._fail("invalid_or_future_observation", now_s)
        if self.state in (InterceptState.EXPIRED, InterceptState.HOLD):
            return self._decision(self.terminal_reason or "manual_reset_required", None, now_s)
        if self.state is InterceptState.COMMITTED:
            if self.last_confirmation_s is None or self.crossing_deadline_s is None:
                return self._fail("committed_state_incomplete", now_s)
            if now_s - self.last_confirmation_s > self.config.perception_ttl_s:
                return self._expire("confirmation_ttl_expired", now_s)
            if now_s > self.crossing_deadline_s + self.config.post_crossing_hold_s:
                return self._expire("crossing_hold_complete", now_s)
        if not self.config.track_matches(observation.class_name, observation.confidence):
            return self._fail("track_class_or_confidence", now_s)
        if now_s - observation.timestamp_s > self.config.perception_ttl_s:
            return self._fail("observation_stale", now_s)
        if (
            self.last_observation_timestamp_s is not None
            and observation.timestamp_s <= self.last_observation_timestamp_s
        ):
            return self._fail("non_monotonic_observation", now_s)
        if self.active_track_id is None:
            self.active_track_id = observation.track_id
        elif observation.track_id != self.active_track_id:
            return self._fail("track_changed", now_s)
        self.last_observation_timestamp_s = observation.timestamp_s
        if observation.consecutive_observations < self.config.minimum_observations:
            if self.latched_target is not None:
                return self._fail("insufficient_observations", now_s)
            self.state = InterceptState.ACQUIRING
            return self._decision("insufficient_observations", None, now_s)
        if observation.residual_m > self.config.maximum_residual_m:
            return self._fail("estimator_residual", now_s)

        plan = plan_plane_intercept(
            object_position_m=observation.position_m,
            object_velocity_m_s=observation.velocity_m_s,
            observation_timestamp_s=observation.timestamp_s,
            now_s=now_s,
            palm_position_m=palm_position_m,
            block_plane_point_m=self.config.plane_point_m,
            block_plane_normal=self.config.plane_normal,
            current_palm_normal=None,
            config=self.config.planner,
        )
        if not plan.ok or plan.target_palm_position_m is None:
            return self._fail(f"planner_{plan.reason}", now_s, plan)
        assert plan.predicted_object_center_m is not None
        assert plan.target_palm_normal is not None
        if not self.config.crossing_inside_bounds(plan.predicted_object_center_m):
            return self._fail("crossing_outside_segment", now_s, plan)
        facing_dot = float(
            np.dot(
                np.asarray(self.config.lane_facing_palm_normal),
                plan.target_palm_normal,
            )
        )
        minimum_dot = math.cos(self.config.planner.maximum_orientation_error_rad)
        if facing_dot < minimum_dot:
            return self._fail("lane_facing_orientation_mismatch", now_s, plan)
        arrival_slack = plan.hold_time_s
        if arrival_slack is None or arrival_slack < self.config.minimum_deadline_slack_s:
            return self._fail("insufficient_deadline_slack", now_s, plan)

        if self.latched_target is not None:
            error = float(np.linalg.norm(plan.target_palm_position_m - self.latched_target))
            if error > self.config.revalidation_tolerance_m:
                return self._fail("committed_target_not_reconfirmed", now_s, plan)
            self.last_confirmation_s = observation.timestamp_s
            if plan.crossing_time_from_now_s is not None:
                self.crossing_deadline_s = now_s + plan.crossing_time_from_now_s
            self.state = InterceptState.COMMITTED
            return self._decision("committed_reconfirmed", plan, now_s)

        crossing_time = plan.crossing_time_from_now_s
        if crossing_time is None or crossing_time > self.config.commit_horizon_s:
            self.state = InterceptState.PREVIEW
            return self._decision("waiting_for_commit_window", plan, now_s)

        self.latched_target = plan.target_palm_position_m.copy()
        self.last_confirmation_s = observation.timestamp_s
        self.crossing_deadline_s = now_s + crossing_time
        self.state = InterceptState.COMMITTED
        return self._decision("committed", plan, now_s)

    def current_without_observation(self, *, now_s: float) -> InterceptDecision:
        if self.state is not InterceptState.COMMITTED:
            if self.state not in (InterceptState.HOLD, InterceptState.EXPIRED):
                self.reset()
            return self._decision(
                self.terminal_reason or "current_detection_missing",
                None,
                now_s,
            )
        if self.last_confirmation_s is None or self.crossing_deadline_s is None:
            return self._fail("committed_state_incomplete", now_s)
        if now_s - self.last_confirmation_s > self.config.perception_ttl_s:
            return self._expire("confirmation_ttl_expired", now_s)
        if now_s > self.crossing_deadline_s + self.config.post_crossing_hold_s:
            return self._expire("crossing_hold_complete", now_s)
        return self._decision("committed_occlusion_hold", None, now_s)

    def _fail(
        self,
        reason: str,
        now_s: float,
        plan: InterceptPlan | None = None,
    ) -> InterceptDecision:
        self.state = InterceptState.HOLD
        self.terminal_reason = reason
        return self._decision(reason, plan, now_s)

    def _expire(self, reason: str, now_s: float) -> InterceptDecision:
        self.state = InterceptState.EXPIRED
        self.terminal_reason = reason
        return self._decision(reason, None, now_s)

    def _decision(
        self,
        reason: str,
        plan: InterceptPlan | None,
        now_s: float,
    ) -> InterceptDecision:
        target_source = self.latched_target
        if target_source is None and plan is not None:
            target_source = plan.target_palm_position_m
        target = (
            None
            if target_source is None
            else tuple(float(value) for value in target_source)
        )
        crossing = (
            None
            if plan is None or plan.predicted_object_center_m is None
            else tuple(float(value) for value in plan.predicted_object_center_m)
        )
        crossing_time = None if plan is None else plan.crossing_time_from_now_s
        if (
            crossing_time is not None
            and (not math.isfinite(crossing_time) or crossing_time < 0.0)
        ):
            # A reversed or already-passed bunny may retain the signed timing
            # in the diagnostic InterceptPlan, but executable command
            # contracts only carry non-negative future durations.
            crossing_time = None
        arrival_slack = None if plan is None else plan.hold_time_s
        if (
            arrival_slack is not None
            and (not math.isfinite(arrival_slack) or arrival_slack < 0.0)
        ):
            arrival_slack = None
        confirmation_age = (
            None
            if self.last_confirmation_s is None
            else max(0.0, now_s - self.last_confirmation_s)
        )
        return InterceptDecision(
            state=self.state,
            reason=reason,
            target_palm_position_m=target,
            predicted_crossing_m=crossing,
            crossing_time_from_now_s=crossing_time,
            arrival_slack_s=arrival_slack,
            last_confirmation_age_s=confirmation_age,
            plan=plan,
        )
