"""Latency- and reachability-aware interception planning.

This module is deliberately independent of ROS, Unitree transport, and Isaac.
It turns a timestamped object position/velocity estimate into a palm contact
pose request.  Collision and joint-limit validation remain the IK controller's
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return value / length


@dataclass(frozen=True)
class InterceptConfig:
    """Physical timing and contact geometry used by the planner."""

    compute_delay_s: float = 0.015
    command_delay_s: float = 0.035
    maximum_palm_speed_m_s: float = 0.35
    maximum_palm_acceleration_m_s2: float = 1.2
    settle_time_s: float = 0.06
    bunny_contact_radius_m: float = 0.055
    palm_half_thickness_m: float = 0.012
    minimum_object_speed_m_s: float = 0.025
    maximum_crossing_horizon_s: float = 3.0
    maximum_orientation_error_rad: float = math.radians(25.0)

    def __post_init__(self) -> None:
        nonnegative = (
            self.compute_delay_s,
            self.command_delay_s,
            self.settle_time_s,
        )
        positive = (
            self.maximum_palm_speed_m_s,
            self.maximum_palm_acceleration_m_s2,
            self.bunny_contact_radius_m,
            self.palm_half_thickness_m,
            self.minimum_object_speed_m_s,
            self.maximum_crossing_horizon_s,
            self.maximum_orientation_error_rad,
        )
        if (
            any(not math.isfinite(value) or value < 0 for value in nonnegative)
            or any(not math.isfinite(value) or value <= 0 for value in positive)
        ):
            raise ValueError("intercept configuration values are invalid")


@dataclass(frozen=True)
class InterceptPlan:
    """A proposed blocking pose and the timing evidence behind it."""

    ok: bool
    reason: str
    target_palm_position_m: np.ndarray | None
    target_palm_normal: np.ndarray | None
    predicted_object_center_m: np.ndarray | None
    crossing_time_from_now_s: float | None
    arm_flight_time_s: float | None
    available_arm_time_s: float | None
    hold_time_s: float | None
    requires_orientation_replan: bool = False

    def to_dict(self) -> dict[str, object]:
        def point(value: np.ndarray | None) -> list[float] | None:
            return None if value is None else value.round(6).tolist()

        return {
            "ok": self.ok,
            "reason": self.reason,
            "target_palm_position_m": point(self.target_palm_position_m),
            "target_palm_normal": point(self.target_palm_normal),
            "predicted_object_center_m": point(self.predicted_object_center_m),
            "crossing_time_from_now_s": self.crossing_time_from_now_s,
            "arm_flight_time_s": self.arm_flight_time_s,
            "available_arm_time_s": self.available_arm_time_s,
            "hold_time_s": self.hold_time_s,
            "requires_orientation_replan": self.requires_orientation_replan,
        }


def minimum_travel_time_s(
    distance_m: float,
    *,
    maximum_speed_m_s: float,
    maximum_acceleration_m_s2: float,
) -> float:
    """Return a conservative triangular/trapezoidal Cartesian travel time."""

    if (
        not math.isfinite(distance_m)
        or distance_m < 0
        or maximum_speed_m_s <= 0
        or maximum_acceleration_m_s2 <= 0
    ):
        raise ValueError("travel-time inputs are invalid")
    acceleration_distance = maximum_speed_m_s**2 / maximum_acceleration_m_s2
    if distance_m <= acceleration_distance:
        return 2.0 * math.sqrt(distance_m / maximum_acceleration_m_s2)
    return (
        2.0 * maximum_speed_m_s / maximum_acceleration_m_s2
        + (distance_m - acceleration_distance) / maximum_speed_m_s
    )


def plan_plane_intercept(
    *,
    object_position_m: Iterable[float],
    object_velocity_m_s: Iterable[float],
    observation_timestamp_s: float,
    now_s: float,
    palm_position_m: Iterable[float],
    block_plane_point_m: Iterable[float],
    block_plane_normal: Iterable[float],
    current_palm_normal: Iterable[float] | None = None,
    config: InterceptConfig | None = None,
) -> InterceptPlan:
    """Plan for the bunny centre to cross a fixed blocking plane.

    The palm is placed ahead of the moving bunny by the bunny contact radius
    plus half the palm thickness.  The returned palm normal faces the oncoming
    object.  A caller must use a full-pose IK solve when
    ``requires_orientation_replan`` is true; translation-only servoing is not
    sufficient in that case.
    """

    cfg = config or InterceptConfig()
    position = _vector3(object_position_m, "object_position_m")
    velocity = _vector3(object_velocity_m_s, "object_velocity_m_s")
    palm = _vector3(palm_position_m, "palm_position_m")
    plane_point = _vector3(block_plane_point_m, "block_plane_point_m")
    plane_normal = _unit(_vector3(block_plane_normal, "block_plane_normal"), "block_plane_normal")
    if (
        not math.isfinite(observation_timestamp_s)
        or not math.isfinite(now_s)
        or observation_timestamp_s < 0
        or now_s < observation_timestamp_s
    ):
        raise ValueError("timestamps are invalid")

    speed = float(np.linalg.norm(velocity))
    if speed < cfg.minimum_object_speed_m_s:
        return InterceptPlan(False, "object_too_slow", None, None, None, None, None, None, None)

    plane_speed = float(np.dot(velocity, plane_normal))
    signed_distance = float(np.dot(position - plane_point, plane_normal))
    if abs(plane_speed) <= 1e-9:
        return InterceptPlan(
            False, "path_does_not_cross_plane", None, None, None, None, None, None, None
        )
    crossing_from_observation_s = -signed_distance / plane_speed
    observation_age_s = now_s - observation_timestamp_s
    crossing_from_now_s = crossing_from_observation_s - observation_age_s
    if crossing_from_now_s <= 0:
        return InterceptPlan(
            False, "crossing_already_passed", None, None, None, crossing_from_now_s, None, None, None
        )
    if crossing_from_now_s > cfg.maximum_crossing_horizon_s:
        return InterceptPlan(
            False, "crossing_too_far", None, None, None, crossing_from_now_s, None, None, None
        )

    predicted_center = position + velocity * crossing_from_observation_s
    travel_direction = velocity / speed
    desired_normal = -travel_direction
    contact_offset_m = cfg.bunny_contact_radius_m + cfg.palm_half_thickness_m
    target_palm = predicted_center + travel_direction * contact_offset_m

    distance_m = float(np.linalg.norm(target_palm - palm))
    arm_flight_s = minimum_travel_time_s(
        distance_m,
        maximum_speed_m_s=cfg.maximum_palm_speed_m_s,
        maximum_acceleration_m_s2=cfg.maximum_palm_acceleration_m_s2,
    ) + cfg.settle_time_s
    available_s = crossing_from_now_s - cfg.compute_delay_s - cfg.command_delay_s
    hold_s = available_s - arm_flight_s
    if available_s <= 0 or hold_s < 0:
        return InterceptPlan(
            False,
            "deadline_unreachable",
            target_palm,
            desired_normal,
            predicted_center,
            crossing_from_now_s,
            arm_flight_s,
            available_s,
            max(0.0, hold_s),
        )

    orientation_replan = False
    if current_palm_normal is not None:
        current_normal = _unit(
            _vector3(current_palm_normal, "current_palm_normal"),
            "current_palm_normal",
        )
        angle = math.acos(float(np.clip(np.dot(current_normal, desired_normal), -1.0, 1.0)))
        orientation_replan = angle > cfg.maximum_orientation_error_rad

    return InterceptPlan(
        True,
        "orientation_replan_required" if orientation_replan else "ok",
        target_palm,
        desired_normal,
        predicted_center,
        crossing_from_now_s,
        arm_flight_s,
        available_s,
        hold_s,
        orientation_replan,
    )
