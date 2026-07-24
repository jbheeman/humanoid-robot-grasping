import math

import numpy as np
import pytest

from object_tracking.intercept_planner import (
    InterceptConfig,
    minimum_travel_time_s,
    plan_plane_intercept,
)


def test_minimum_travel_time_uses_triangular_and_trapezoidal_profiles() -> None:
    assert minimum_travel_time_s(
        0.04, maximum_speed_m_s=0.4, maximum_acceleration_m_s2=1.0
    ) == pytest.approx(0.4)
    assert minimum_travel_time_s(
        0.32, maximum_speed_m_s=0.4, maximum_acceleration_m_s2=1.0
    ) == pytest.approx(1.2)


def test_plan_places_palm_ahead_of_bunny_and_faces_it() -> None:
    plan = plan_plane_intercept(
        object_position_m=(0.30, 0.20, 0.10),
        object_velocity_m_s=(0.0, -0.10, 0.0),
        observation_timestamp_s=10.0,
        now_s=10.0,
        palm_position_m=(0.30, -0.06, 0.10),
        block_plane_point_m=(0.30, 0.0, 0.10),
        block_plane_normal=(0.0, 1.0, 0.0),
        current_palm_normal=(0.0, 1.0, 0.0),
        config=InterceptConfig(
            maximum_palm_speed_m_s=0.5,
            maximum_palm_acceleration_m_s2=2.0,
            settle_time_s=0.02,
        ),
    )
    assert plan.ok
    assert plan.reason == "ok"
    assert plan.crossing_time_from_now_s == pytest.approx(2.0)
    assert np.allclose(plan.predicted_object_center_m, (0.30, 0.0, 0.10))
    assert np.allclose(plan.target_palm_normal, (0.0, 1.0, 0.0))
    assert plan.target_palm_position_m[1] < 0.0
    assert plan.hold_time_s > 0


def test_stale_observation_can_make_intercept_unreachable() -> None:
    plan = plan_plane_intercept(
        object_position_m=(0.30, 0.05, 0.10),
        object_velocity_m_s=(0.0, -0.10, 0.0),
        observation_timestamp_s=10.0,
        now_s=10.4,
        palm_position_m=(0.30, -0.30, 0.10),
        block_plane_point_m=(0.30, 0.0, 0.10),
        block_plane_normal=(0.0, 1.0, 0.0),
    )
    assert not plan.ok
    assert plan.reason == "deadline_unreachable"


def test_orientation_mismatch_requires_full_pose_replan() -> None:
    plan = plan_plane_intercept(
        object_position_m=(0.30, 0.20, 0.10),
        object_velocity_m_s=(0.0, -0.10, 0.0),
        observation_timestamp_s=10.0,
        now_s=10.0,
        palm_position_m=(0.30, -0.06, 0.10),
        block_plane_point_m=(0.30, 0.0, 0.10),
        block_plane_normal=(0.0, 1.0, 0.0),
        current_palm_normal=(1.0, 0.0, 0.0),
        config=InterceptConfig(maximum_orientation_error_rad=math.radians(20.0)),
    )
    assert plan.ok
    assert plan.requires_orientation_replan
    assert plan.reason == "orientation_replan_required"
