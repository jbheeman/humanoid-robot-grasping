from dataclasses import replace
from pathlib import Path

import pytest

from object_tracking.arm_tracking.interception import (
    InterceptObservation,
    InterceptState,
    LiveInterceptConfig,
    LiveInterceptController,
    load_live_intercept_config,
)
from object_tracking.intercept_planner import InterceptConfig


def config(**changes: object) -> LiveInterceptConfig:
    base = LiveInterceptConfig(
        calibration_id="calibration-a",
        plane_point_m=(0.3, 0.0, 0.1),
        plane_normal=(0.0, 1.0, 0.0),
        crossing_minimum_m=(0.2, -0.01, 0.05),
        crossing_maximum_m=(0.4, 0.01, 0.2),
        wanted_classes=("bunny", "plush"),
        minimum_confidence=0.5,
        ready_right_arm_q_rad=(0.0,) * 7,
        ready_pose_tolerance_rad=0.05,
        lane_facing_palm_normal=(0.0, 1.0, 0.0),
        minimum_observations=3,
        maximum_residual_m=0.03,
        commit_horizon_s=0.5,
        post_crossing_hold_s=0.1,
        revalidation_tolerance_m=0.02,
        minimum_deadline_slack_s=0.05,
        perception_ttl_s=0.25,
        planner=InterceptConfig(
            compute_delay_s=0.01,
            command_delay_s=0.01,
            maximum_palm_speed_m_s=0.5,
            maximum_palm_acceleration_m_s2=2.0,
            settle_time_s=0.02,
            maximum_crossing_horizon_s=3.0,
        ),
    )
    return replace(base, **changes)


def observation(
    *,
    timestamp_s: float = 10.0,
    position_x: float = 0.3,
    position_y: float = 0.04,
    velocity_m_s: tuple[float, float, float] = (0.0, -0.1, 0.0),
    track_id: int = 1,
    consecutive: int = 3,
    residual_m: float = 0.005,
) -> InterceptObservation:
    return InterceptObservation(
        track_id=track_id,
        class_name="white bunny",
        confidence=0.9,
        position_m=(position_x, position_y, 0.1),
        velocity_m_s=velocity_m_s,
        timestamp_s=timestamp_s,
        consecutive_observations=consecutive,
        residual_m=residual_m,
    )


def test_preview_then_commit_latches_cartesian_target() -> None:
    controller = LiveInterceptController(config())

    preview = controller.update(
        observation(position_y=0.2),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )
    assert preview.state is InterceptState.PREVIEW
    assert not preview.may_publish

    committed = controller.update(
        observation(timestamp_s=10.1, position_y=0.04),
        now_s=10.1,
        palm_position_m=(0.3, -0.06, 0.1),
    )
    assert committed.state is InterceptState.COMMITTED
    assert committed.may_publish
    target = committed.target_palm_position_m

    reconfirmed = controller.update(
        observation(timestamp_s=10.2, position_y=0.03),
        now_s=10.2,
        palm_position_m=(0.3, -0.06, 0.1),
    )
    assert reconfirmed.state is InterceptState.COMMITTED
    assert reconfirmed.target_palm_position_m == target
    assert reconfirmed.reason == "committed_reconfirmed"


def test_commit_expires_after_last_confirmation_ttl() -> None:
    controller = LiveInterceptController(config())
    controller.update(
        observation(),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )

    held = controller.current_without_observation(now_s=10.20)
    expired = controller.current_without_observation(now_s=10.251)

    assert held.state is InterceptState.COMMITTED
    assert held.reason == "committed_occlusion_hold"
    assert expired.state is InterceptState.EXPIRED
    assert not expired.may_publish


def test_fresh_detection_cannot_resurrect_commit_after_confirmation_ttl() -> None:
    controller = LiveInterceptController(config())
    controller.update(
        observation(),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )

    decision = controller.update(
        observation(timestamp_s=10.3, position_y=0.01),
        now_s=10.3,
        palm_position_m=(0.3, -0.06, 0.1),
    )

    assert decision.state is InterceptState.EXPIRED
    assert decision.reason == "confirmation_ttl_expired"
    assert not decision.may_publish


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"track_id": 2, "timestamp_s": 10.1}, "track_changed"),
        ({"timestamp_s": 10.1, "residual_m": 0.04}, "estimator_residual"),
    ],
)
def test_invalid_reconfirmation_enters_terminal_hold(
    changed: dict[str, object],
    reason: str,
) -> None:
    controller = LiveInterceptController(config())
    controller.update(
        observation(),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )
    invalid = replace(observation(), **changed)

    decision = controller.update(
        invalid,
        now_s=float(changed.get("timestamp_s", 10.1)),
        palm_position_m=(0.3, -0.06, 0.1),
    )

    assert decision.state is InterceptState.HOLD
    assert decision.reason == reason
    assert not decision.may_publish


def test_insufficient_history_never_commits() -> None:
    controller = LiveInterceptController(config())

    decision = controller.update(
        observation(consecutive=2),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )

    assert decision.state is InterceptState.ACQUIRING
    assert decision.reason == "insufficient_observations"
    assert not decision.may_publish


def test_config_loader_is_versioned_and_execution_marked(tmp_path: Path) -> None:
    path = tmp_path / "intercept.yaml"
    path.write_text(
        """
schema_version: 1
calibration_id: calibration-a
plane_point_m: [0.3, 0.0, 0.1]
plane_normal: [0.0, 1.0, 0.0]
crossing_minimum_m: [0.2, -0.01, 0.05]
crossing_maximum_m: [0.4, 0.01, 0.2]
wanted_classes: [bunny]
minimum_confidence: 0.5
ready_right_arm_q_rad: [0, 0, 0, 0, 0, 0, 0]
ready_pose_tolerance_rad: 0.05
lane_facing_palm_normal: [0, 1, 0]
minimum_observations: 3
maximum_residual_m: 0.03
commit_horizon_s: 0.5
post_crossing_hold_s: 0.1
revalidation_tolerance_m: 0.02
minimum_deadline_slack_s: 0.05
perception_ttl_s: 0.25
validated_for_execution: false
planner:
  compute_delay_s: 0.01
  command_delay_s: 0.01
  maximum_palm_speed_m_s: 0.5
  maximum_palm_acceleration_m_s2: 2.0
  settle_time_s: 0.02
  bunny_contact_radius_m: 0.055
  palm_half_thickness_m: 0.012
  minimum_object_speed_m_s: 0.025
  maximum_crossing_horizon_s: 3.0
  maximum_orientation_error_rad: 0.436332
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_live_intercept_config(path)

    assert loaded.calibration_id == "calibration-a"
    assert loaded.validated_for_execution is False


def test_config_rejects_ttl_above_robot_limit() -> None:
    with pytest.raises(ValueError, match="500 ms"):
        config(perception_ttl_s=0.501)


@pytest.mark.parametrize(
    ("observation_changes", "config_changes", "now_s", "reason"),
    [
        ({"velocity_m_s": (0.0, 0.0, 0.0)}, {}, 10.0, "planner_object_too_slow"),
        ({"position_m": (0.3, -0.01, 0.1)}, {}, 10.0, "planner_crossing_already_passed"),
        ({}, {}, 10.251, "observation_stale"),
        (
            {"velocity_m_s": (0.1, 0.0, 0.0)},
            {},
            10.0,
            "planner_path_does_not_cross_plane",
        ),
        ({"position_m": (0.5, 0.04, 0.1)}, {}, 10.0, "crossing_outside_segment"),
        (
            {},
            {"lane_facing_palm_normal": (1.0, 0.0, 0.0)},
            10.0,
            "lane_facing_orientation_mismatch",
        ),
        ({}, {"minimum_deadline_slack_s": 0.4}, 10.0, "insufficient_deadline_slack"),
    ],
)
def test_invalid_intercept_evidence_enters_hold(
    observation_changes: dict[str, object],
    config_changes: dict[str, object],
    now_s: float,
    reason: str,
) -> None:
    controller = LiveInterceptController(config(**config_changes))
    candidate = replace(observation(), **observation_changes)

    decision = controller.update(
        candidate,
        now_s=now_s,
        palm_position_m=(candidate.position_m[0], -0.06, 0.1),
    )

    assert decision.state is InterceptState.HOLD
    assert decision.reason == reason
    assert not decision.may_publish


def test_missing_detection_before_commit_resets_to_acquiring_without_target() -> None:
    controller = LiveInterceptController(config())
    preview = controller.update(
        observation(position_y=0.2),
        now_s=10.0,
        palm_position_m=(0.3, -0.06, 0.1),
    )
    assert preview.state is InterceptState.PREVIEW

    missing = controller.current_without_observation(now_s=10.1)

    assert missing.state is InterceptState.ACQUIRING
    assert missing.target_palm_position_m is None
    assert not missing.may_publish


def test_config_loader_rejects_string_execution_marker(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    test_config_loader_is_versioned_and_execution_marked(tmp_path)
    # The helper above writes intercept.yaml in the same directory.
    source = tmp_path / "intercept.yaml"
    path.write_text(
        source.read_text(encoding="utf-8").replace(
            "validated_for_execution: false",
            'validated_for_execution: "false"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        load_live_intercept_config(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"plane_normal": (0.0, 2.0, 0.0)},
        {
            "crossing_minimum_m": (0.5, 0.0, 0.0),
            "crossing_maximum_m": (0.4, 0.1, 0.1),
        },
        {"wanted_classes": ()},
        {"ready_right_arm_q_rad": (0.0,) * 6},
        {"post_crossing_hold_s": 0.251},
        {"commit_horizon_s": 3.1},
    ],
)
def test_live_config_rejects_invalid_geometry_and_windows(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        config(**changes)
