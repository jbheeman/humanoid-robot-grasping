from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pytest
import yaml

from object_tracking.arm_tracking import runtime as runtime_module
from object_tracking.arm_tracking.calibration import (
    Calibration,
    CalibrationPose,
    RegistrationResiduals,
    ResidualSummary,
    StreamProfile,
    save_calibration_atomic,
)
from object_tracking.arm_tracking.depth import DepthEstimate, DepthFrame
from object_tracking.arm_tracking.geometry import (
    CameraIntrinsics,
    Plane,
    RigidTransform,
    SupportRegion,
    WorkspaceBounds,
)
from object_tracking.arm_tracking.runtime import (
    ArmTrackingRuntime,
    RuntimeConfig,
    clamp_point_height_to_support,
    measured_right_arm_for_intercept,
    register_depth_in_rgb,
    right_arm_ik_seed,
    select_start_escape_waypoint,
)


class FakeTrackingTransport:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.stops: list[str] = []
        self.state: dict[str, Any] = {"state": "DISARMED", "weight": 0.0}
        self.depth_frames: list[DepthFrame] = []
        self.published: list[dict[str, Any]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def receive_depth(self, timeout_s: float) -> DepthFrame | None:
        del timeout_s
        return self.depth_frames.pop(0) if self.depth_frames else None

    def arm_state(self) -> dict[str, Any]:
        return dict(self.state)

    def enable_arm(self, session_id: str, calibration_id: str) -> dict[str, Any]:
        return {"state": "ARMED", "session_id": session_id, "calibration_id": calibration_id}

    def publish_target(
        self,
        session_id: str,
        sequence: int,
        calibration_id: str,
        right_arm_q: Sequence[float],
        pipeline_age_ms: float,
    ) -> None:
        self.published.append(
            {
                "session_id": session_id,
                "sequence": sequence,
                "calibration_id": calibration_id,
                "right_arm_q": tuple(float(value) for value in right_arm_q),
                "pipeline_age_ms": pipeline_age_ms,
            }
        )

    def stop_arm(self, reason: str) -> None:
        self.stops.append(reason)

    def commissioning(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"command": command, **payload}


def _calibration() -> Calibration:
    profile = StreamProfile(4, 3, 30, "z16")
    intrinsics = CameraIntrinsics(4, 3, 2.0, 2.0, 1.5, 1.0)
    residuals = ResidualSummary(0.0, 0.0, 0.0)
    solve_poses = tuple(
        CalibrationPose(
            f"solve-{index}",
            (float(index % 4), float(index // 4), 1.0),
            (float(index % 4), float(index // 4), 1.0),
            0.0,
            0.0,
        )
        for index in range(8)
    )
    validation_poses = tuple(
        CalibrationPose(
            f"validation-{index}",
            (float(index % 2) + 0.25, float(index // 2) + 0.25, 1.0),
            (float(index % 2) + 0.25, float(index // 2) + 0.25, 1.0),
            0.0,
            0.0,
        )
        for index in range(4)
    )
    calibration = Calibration(
        calibration_id="00000000-0000-0000-0000-000000000001",
        calibration_hash="pending",
        created_at="2026-01-01T00:00:00Z",
        camera_serial="serial",
        camera_firmware="firmware",
        rgb_profile=replace(profile, format="rgb8"),
        depth_profile=profile,
        rgb_intrinsics=intrinsics,
        depth_intrinsics=intrinsics,
        depth_scale=0.001,
        depth_to_rgb=RigidTransform.identity(),
        optical_to_torso=RigidTransform.identity(),
        waist_reference_rad=(0.0, 0.0, 0.0),
        tag_to_wrist=RigidTransform.identity(),
        workspace=WorkspaceBounds((-2, -2, 0), (2, 2, 3)),
        solve_poses=solve_poses,
        validation_poses=validation_poses,
        solve_residuals=residuals,
        validation_residuals=residuals,
        registration_residuals=RegistrationResiduals(0.0, 0.0, 8),
    )
    return calibration.with_computed_hash()


def _write_intercept_profile(path: Path, calibration_id: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "calibration_id": calibration_id,
                "plane_point_m": [0.3, 0.0, 0.12],
                "plane_normal": [0.0, 1.0, 0.0],
                "crossing_minimum_m": [0.2, -0.01, 0.1],
                "crossing_maximum_m": [0.4, 0.01, 0.2],
                "wanted_classes": ["bunny"],
                "minimum_confidence": 0.5,
                "ready_right_arm_q_rad": [0.0] * 7,
                "ready_pose_tolerance_rad": 0.05,
                "lane_facing_palm_normal": [0.0, 1.0, 0.0],
                "minimum_observations": 3,
                "maximum_residual_m": 0.03,
                "commit_horizon_s": 0.5,
                "post_crossing_hold_s": 0.1,
                "revalidation_tolerance_m": 0.02,
                "minimum_deadline_slack_s": 0.05,
                "perception_ttl_s": 0.25,
                "validated_for_execution": True,
                "planner": {
                    "compute_delay_s": 0.01,
                    "command_delay_s": 0.01,
                    "maximum_palm_speed_m_s": 0.5,
                    "maximum_palm_acceleration_m_s2": 2.0,
                    "settle_time_s": 0.02,
                    "bunny_contact_radius_m": 0.055,
                    "palm_half_thickness_m": 0.012,
                    "minimum_object_speed_m_s": 0.025,
                    "maximum_crossing_horizon_s": 3.0,
                    "maximum_orientation_error_rad": 0.436332,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class FakeInterceptIK:
    global_backend = "fake"
    local_backend = "fake"

    def forward_kinematics(self, right_arm_q: Sequence[float]) -> np.ndarray:
        del right_arm_q
        transform = np.eye(4)
        transform[:3, 3] = (0.3, -0.067, 0.12)
        return transform

    def collision_labels(self, right_arm_q: Sequence[float]) -> tuple[str, ...]:
        del right_arm_q
        return ()

    def solve_local_translation(
        self,
        target_transform: np.ndarray,
        initial_q: Sequence[float],
        *,
        support_plane: SupportRegion,
    ) -> Any:
        del target_transform, support_plane
        from object_tracking.arm_tracking.ik_solver import IKResult

        return IKResult(True, tuple(float(value) for value in initial_q), 0.0, 0.0)


def test_identity_registration_preserves_valid_depth_pixels() -> None:
    z16 = np.zeros((3, 4), dtype=np.uint16)
    z16[1, 2] = 1000
    aligned = register_depth_in_rgb(z16, _calibration())
    assert aligned.shape == z16.shape
    assert aligned[1, 2] == 1000


def test_registered_point_height_is_projected_onto_safe_table_height() -> None:
    plane = Plane((0.0, 0.0, 1.0), 0.0)

    corrected, height = clamp_point_height_to_support((0.52, 0.14, -0.05), plane)

    np.testing.assert_allclose(corrected, (0.52, 0.14, 0.03))
    assert height == 0.03


def test_registered_point_height_accepts_bounded_support_region() -> None:
    support = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (0.35, -0.35),
        (0.80, 0.35),
    )

    corrected, height = clamp_point_height_to_support(
        (0.52, 0.14, -0.05),
        support,
    )

    np.testing.assert_allclose(corrected, (0.52, 0.14, 0.03))
    assert height == 0.03


def test_ik_seed_prefers_fresh_measured_right_arm_over_disarmed_command() -> None:
    measured = [float(index) for index in range(29)]
    state = {
        "commanded_arm_q": [0.0] * 14,
        "visualization": {
            "available": True,
            "state_age_ms": 12.0,
            "measured_pose_rad": measured,
        },
    }

    assert right_arm_ik_seed(state, None) == measured[22:29]


def test_intercept_requires_fresh_measured_right_arm() -> None:
    measured = [float(index) for index in range(29)]
    state = {
        "visualization": {
            "available": True,
            "state_age_ms": 12.0,
            "measured_pose_rad": measured,
        }
    }

    right, error = measured_right_arm_for_intercept(state)

    assert right == tuple(measured[22:29])
    assert error is None


def test_intercept_never_falls_back_to_commanded_arm() -> None:
    state = {
        "commanded_arm_q": [0.0] * 14,
        "visualization": {
            "available": True,
            "state_age_ms": 251.0,
            "measured_pose_rad": [0.0] * 29,
        },
    }

    right, error = measured_right_arm_for_intercept(state)

    assert right is None
    assert error == "measured_arm_stale"


@pytest.mark.parametrize(
    ("visualization", "error"),
    [
        ({"available": False}, "measured_arm_unavailable"),
        (
            {"available": True, "measured_pose_rad": [0.0] * 29},
            "measured_arm_age_unavailable",
        ),
        (
            {
                "available": True,
                "state_age_ms": 1.0,
                "measured_pose_rad": [0.0] * 28,
            },
            "measured_arm_shape",
        ),
        (
            {
                "available": True,
                "state_age_ms": 1.0,
                "measured_pose_rad": [0.0] * 28 + [float("nan")],
            },
            "measured_arm_non_finite",
        ),
    ],
)
def test_intercept_rejects_missing_or_malformed_measured_arm(
    visualization: dict[str, Any],
    error: str,
) -> None:
    right, actual_error = measured_right_arm_for_intercept(
        {
            "commanded_arm_q": [0.0] * 14,
            "visualization": visualization,
        }
    )

    assert right is None
    assert actual_error == error


def test_intercept_profile_must_match_camera_calibration(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    intercept_path = tmp_path / "intercept.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    _write_intercept_profile(intercept_path, "different-calibration")

    with pytest.raises(ValueError, match="does not match"):
        ArmTrackingRuntime(
            RuntimeConfig(
                calibration_path=calibration_path,
                intercept_config_path=intercept_path,
            ),
            lambda: {},
            lambda status, depth: None,
            transport=FakeTrackingTransport(),
            repo_root=tmp_path,
        )


def test_execute_rejects_intercept_profile_not_physically_validated(
    tmp_path: Path,
) -> None:
    calibration = _calibration()
    calibration_path = tmp_path / "calibration.yaml"
    intercept_path = tmp_path / "intercept.yaml"
    save_calibration_atomic(calibration, calibration_path)
    _write_intercept_profile(intercept_path, calibration.calibration_id)
    profile = yaml.safe_load(intercept_path.read_text(encoding="utf-8"))
    profile["validated_for_execution"] = False
    intercept_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="not validated_for_execution"):
        ArmTrackingRuntime(
            RuntimeConfig(
                calibration_path=calibration_path,
                intercept_config_path=intercept_path,
                execute=True,
            ),
            lambda: {},
            lambda status, depth: None,
            transport=FakeTrackingTransport(),
            repo_root=tmp_path,
        )


def test_start_escape_waypoint_uses_one_knot_measured_lookahead() -> None:
    path = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.08, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    waypoint, index, error = select_start_escape_waypoint(path[0], path, 1)
    assert waypoint == path[2]
    assert index == 2
    assert error is None

    waypoint, index, error = select_start_escape_waypoint(path[1], path, 1)
    assert waypoint == path[2]
    assert index == 2
    assert error is None


def test_start_escape_waypoint_rejects_tracking_drift() -> None:
    path = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    waypoint, index, error = select_start_escape_waypoint(
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        path,
        1,
    )

    assert waypoint is None
    assert index == 1
    assert error == "escape_path_tracking_error"


def test_start_escape_waypoint_accepts_asynchronous_joint_progress() -> None:
    path = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.016),
    )

    waypoint, index, error = select_start_escape_waypoint(
        (0.0, -0.005, 0.0, 0.0, 0.0, 0.0, 0.024),
        path,
        1,
        reached_tolerance_rad=0.015,
    )

    assert waypoint == path[1]
    assert index == 1
    assert error is None


def test_start_escape_waypoint_advances_past_observed_servo_residual() -> None:
    path = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.065, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    waypoint, index, error = select_start_escape_waypoint(
        (0.0, -0.02468, 0.0, 0.0, 0.0, 0.0, 0.0),
        path,
        1,
    )

    assert waypoint == path[2]
    assert index == 2
    assert error is None


def test_start_escape_waypoint_advances_past_larger_loaded_servo_residual() -> None:
    path = (
        (0.0, -0.30, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.34, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.38, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    waypoint, index, error = select_start_escape_waypoint(
        (0.0, -0.31377, 0.0, 0.0, 0.0, 0.0, 0.0),
        path,
        1,
    )

    assert waypoint == path[2]
    assert index == 2
    assert error is None


def test_runtime_uses_injected_transport_for_arm_state_and_stop(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    transport = FakeTrackingTransport()
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path, execute=True),
        lambda: {},
        lambda status, depth: None,
        transport=transport,
        repo_root=tmp_path,
    )

    assert runtime._arm_state() == {"state": "DISARMED", "weight": 0.0}
    runtime.last_target_track = 3
    runtime._approach_path = ((0.0,) * 7, (0.01,) * 7)
    runtime._approach_target_index = 1
    runtime._approach_target_xyz = (0.4, 0.0, 0.2)
    runtime._stop_arm("test_stop")
    assert transport.stops == ["test_stop"]
    assert runtime._approach_path is None
    assert runtime._approach_target_index == 1
    assert runtime._approach_target_xyz is None


def test_transient_perception_rejection_leaves_session_for_deadman(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    transport = FakeTrackingTransport()
    statuses: list[dict[str, object]] = []
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path, execute=True),
        lambda: {},
        lambda status, depth: statuses.append(status),
        transport=transport,
        repo_root=tmp_path,
    )
    runtime.last_target_track = 3
    runtime._approach_path = ((0.0,) * 7, (0.01,) * 7)

    runtime._reject({}, "rgb_depth_pair_stale", None)

    assert transport.stops == []
    assert runtime.last_target_track == 3
    assert runtime._approach_path is not None
    assert statuses[-1]["reason"] == "rgb_depth_pair_stale"


def test_runtime_config_has_no_http_or_token_requirement(tmp_path: Path) -> None:
    config = RuntimeConfig(calibration_path=tmp_path / "calibration.yaml", execute=True)
    assert config.execute is True
    assert config.max_pair_skew_s == pytest.approx(0.240)
    assert not hasattr(config, "depth_ws_url")
    assert not hasattr(config, "arm_url")
    assert not hasattr(config, "arm_token_file")


def test_visualization_reports_cached_support_during_pair_miss(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path),
        lambda: {},
        lambda status, depth: None,
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )
    support = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), -0.1),
        (0.0, 0.0),
        (1.0, 1.0),
        source="test_cached_plane",
    )
    runtime.last_support_plane = support
    runtime.last_support_plane_at = time.monotonic()

    visualization = runtime._visualization_context({"state": "DISARMED"})

    assert visualization["support_plane"] == support.to_dict()
    status = visualization["support_plane_status"]
    assert status["available"] is True
    assert status["cached"] is True
    assert status["source"] == "test_cached_plane"


def test_runtime_receives_depth_from_injected_transport(tmp_path: Path) -> None:
    calibration = _calibration()
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(calibration, calibration_path)
    transport = FakeTrackingTransport()
    transport.depth_frames.append(
        DepthFrame(
            sequence=7,
            receipt_time_s=1.0,
            z16=np.ones((3, 4), dtype=np.uint16),
            depth_scale=0.001,
            calibration_id=calibration.calibration_id,
        )
    )
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path),
        lambda: {},
        lambda status, depth: None,
        transport=transport,
        repo_root=tmp_path,
    )
    runtime._process_latest = runtime.stop_event.set  # type: ignore[method-assign]

    runtime._run()

    assert transport.started is True
    assert runtime.last_depth is not None
    assert runtime.last_depth.sequence == 7
    runtime.stop()
    assert transport.closed is True


def test_runtime_reuses_recent_support_plane_without_refitting(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path),
        lambda: {},
        lambda status, depth: None,
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )
    cached = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (0.35, -0.35),
        (0.80, 0.35),
    )
    runtime.last_support_plane = cached
    runtime.last_support_plane_at = time.monotonic()

    result = runtime._support_plane(np.ones((3, 4), dtype=np.uint16), 0.001)

    assert result is cached
    assert runtime.last_support_plane_error is None


def test_runtime_rescales_clicked_corners_to_active_rgb_profile(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    tabletop_path = tmp_path / "tabletop.json"
    save_calibration_atomic(_calibration(), calibration_path)
    tabletop_path.write_text(
        json.dumps(
            {
                "camera_frame": {"width": 8, "height": 6},
                "tabletop": {"depth_m": 0.4, "width_m": 0.6},
                "corners_px": [
                    {"name": "near_left", "x": 2, "y": 4},
                    {"name": "near_right", "x": 6, "y": 4},
                    {"name": "far_right", "x": 6, "y": 2},
                    {"name": "far_left", "x": 2, "y": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime = ArmTrackingRuntime(
        RuntimeConfig(
            calibration_path=calibration_path,
            tabletop_path=tabletop_path,
            automatic_support_plane=False,
        ),
        lambda: {},
        lambda status, depth: None,
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )

    assert runtime._scaled_tabletop_corners() == (
        (1.0, 2.0),
        (3.0, 2.0),
        (3.0, 1.0),
        (1.0, 1.0),
    )


def test_runtime_can_use_explicit_nominal_plane_without_table(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    calibration = _calibration()
    save_calibration_atomic(calibration, calibration_path)
    runtime = ArmTrackingRuntime(
        RuntimeConfig(
            calibration_path=calibration_path,
            allow_nominal_support_plane=True,
        ),
        lambda: {},
        lambda status, depth: None,
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )

    support = runtime._support_plane(np.zeros((3, 4), dtype=np.uint16), 0.001)

    assert support is not None
    assert support.source == "calibrated_nominal_free_space_plane"
    assert support.plane.signed_distance(
        (0.0, 0.0, runtime._expected_table_height_m())
    ) == 0.0


def test_runtime_freezes_validated_plane_during_armed_motion(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(_calibration(), calibration_path)
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path),
        lambda: {},
        lambda status, depth: None,
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )
    cached = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (0.35, -0.35),
        (0.80, 0.35),
    )
    runtime.last_support_plane = cached
    runtime.last_support_plane_at = time.monotonic()

    result = runtime._support_plane(
        np.ones((3, 4), dtype=np.uint16),
        0.001,
        freeze=True,
    )

    assert result is cached
    assert runtime.last_support_plane_error is None


def test_continuous_mode_target_loss_order_is_unchanged_without_intercept_profile(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calibration = _calibration()
    calibration_path = tmp_path / "calibration.yaml"
    save_calibration_atomic(calibration, calibration_path)
    receipt_time = time.monotonic()
    statuses: list[dict[str, Any]] = []
    runtime = ArmTrackingRuntime(
        RuntimeConfig(calibration_path=calibration_path),
        lambda: {
            "rgb_receipt_time_s": receipt_time,
            "rgb_shape": (3, 4, 3),
            "rgb_frame_id": 1,
            "tracks": [],
        },
        lambda status, depth: statuses.append(dict(status)),
        transport=FakeTrackingTransport(),
        repo_root=tmp_path,
    )
    runtime.last_depth = DepthFrame(
        sequence=1,
        receipt_time_s=receipt_time,
        z16=np.ones((3, 4), dtype=np.uint16),
        depth_scale=0.001,
        calibration_id=calibration.calibration_id,
        registered_to_rgb=True,
    )
    monkeypatch.setattr(runtime_module, "depth_colormap_jpeg", lambda z16, scale: None)
    monkeypatch.setattr(
        runtime,
        "_support_plane",
        lambda *args, **kwargs: pytest.fail("support plane should not run before target loss"),
    )

    runtime._process_latest()

    assert statuses[-1]["reason"] == "target_lost"


@pytest.mark.parametrize("execute", [False, True])
def test_intercept_preview_and_commit_publish_gate(
    tmp_path: Path,
    monkeypatch: Any,
    execute: bool,
) -> None:
    calibration = _calibration()
    calibration_path = tmp_path / "calibration.yaml"
    intercept_path = tmp_path / "intercept.yaml"
    save_calibration_atomic(calibration, calibration_path)
    _write_intercept_profile(intercept_path, calibration.calibration_id)

    clock = [100.0]
    object_y = [0.06]
    statuses: list[dict[str, Any]] = []
    transport = FakeTrackingTransport()
    transport.state = {
        "state": "ARMED",
        "session_id": "intercept-test",
        "weight": 1.0,
        "visualization": {
            "available": True,
            "state_age_ms": 5.0,
            "measured_pose_rad": [0.0] * 29,
        },
    }
    def snapshot() -> dict[str, Any]:
        return {
            "rgb_receipt_time_s": clock[0],
            "rgb_shape": (3, 4, 3),
            "rgb_frame_id": int(round((clock[0] - 100.0) * 10.0)) + 1,
            "inference_started_monotonic_s": clock[0],
            "inference_completed_monotonic_s": clock[0],
            "tracks": [
                {
                    "track_id": 7,
                    "class_name": "white bunny",
                    "confidence": 0.95,
                    "bbox_xyxy": [0.0, 0.0, 3.0, 2.0],
                    "missed_updates": 0,
                }
            ],
        }
    runtime = ArmTrackingRuntime(
        RuntimeConfig(
            calibration_path=calibration_path,
            execute=execute,
            intercept_config_path=intercept_path,
        ),
        snapshot,
        lambda status, depth: statuses.append(dict(status)),
        transport=transport,
        repo_root=tmp_path,
    )
    runtime.ik = FakeInterceptIK()  # type: ignore[assignment]
    support = SupportRegion.from_xy_bounds(
        Plane((0.0, 0.0, 1.0), 0.0),
        (-0.5, -0.5),
        (0.8, 0.5),
    )

    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_module, "depth_colormap_jpeg", lambda z16, scale: None)
    monkeypatch.setattr(runtime, "_support_plane", lambda *args, **kwargs: support)
    monkeypatch.setattr(
        runtime_module,
        "estimate_adaptive_roi_depth",
        lambda *args, **kwargs: DepthEstimate(1.0, (1.5, 1.0), 12, 1.0, 0.001, 0.002),
    )
    monkeypatch.setattr(
        runtime_module,
        "clamp_point_height_to_support",
        lambda point, plane: (np.asarray((0.3, object_y[0], 0.12)), 0.12),
    )

    for index, y_m in enumerate((0.06, 0.05, 0.04, 0.03, 0.02)):
        clock[0] = 100.0 + index * 0.1
        object_y[0] = y_m
        runtime.last_depth = DepthFrame(
            sequence=index + 1,
            receipt_time_s=clock[0],
            z16=np.ones((3, 4), dtype=np.uint16),
            depth_scale=0.001,
            calibration_id=calibration.calibration_id,
            registered_to_rgb=True,
        )
        runtime._process_latest()
        if index < 4:
            assert transport.published == []

    assert len(transport.published) == int(execute)
    if execute:
        assert transport.published[0]["session_id"] == "intercept-test"
        assert statuses[-1]["status"] == "target_sent"
    else:
        assert statuses[-1]["status"] == "tracking"
    assert statuses[-1]["intercept"]["state"] == "COMMITTED"
    assert statuses[-1]["next_edge_collision_checked"] is True
    assert statuses[-1]["ik_step_type"] == "intercept_local_translation"
    assert statuses[-1]["timing_clock_domain"] == "gb10_monotonic_receipt"
    assert statuses[-1]["localization_filter_latency_ms"] is not None
    assert statuses[-1]["intercept_planning_latency_ms"] is not None
    assert statuses[-1]["ik_latency_ms"] is not None
    assert (statuses[-1]["target_publish_latency_ms"] is not None) is execute
    assert statuses[-1]["estimator_consecutive_observations"] == 5
    assert statuses[-1]["estimator_residual_m"] > 0.0

    runtime.reset_intercept()

    assert transport.stops == (["intercept_reset"] if execute else [])
    assert runtime.intercept_controller is not None
    assert runtime.intercept_controller.state.value == "ACQUIRING"
