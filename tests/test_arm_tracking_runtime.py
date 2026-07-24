from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from object_tracking.arm_tracking.calibration import (
    Calibration,
    CalibrationPose,
    RegistrationResiduals,
    ResidualSummary,
    StreamProfile,
    save_calibration_atomic,
)
from object_tracking.arm_tracking.depth import DepthFrame
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
    register_depth_in_rgb,
    right_arm_ik_seed,
)


class FakeTrackingTransport:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.stops: list[str] = []
        self.state: dict[str, Any] = {"state": "DISARMED", "weight": 0.0}
        self.depth_frames: list[DepthFrame] = []

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
        del session_id, sequence, calibration_id, right_arm_q, pipeline_age_ms

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


def test_ik_seed_uses_measured_right_arm_when_no_command_exists() -> None:
    measured = [float(index) for index in range(29)]
    state = {"visualization": {"measured_pose_rad": measured}}

    assert right_arm_ik_seed(state, None) == measured[22:29]


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
    runtime._stop_arm("test_stop")
    assert transport.stops == ["test_stop"]


def test_runtime_config_has_no_http_or_token_requirement(tmp_path: Path) -> None:
    config = RuntimeConfig(calibration_path=tmp_path / "calibration.yaml", execute=True)
    assert config.execute is True
    assert not hasattr(config, "depth_ws_url")
    assert not hasattr(config, "arm_url")
    assert not hasattr(config, "arm_token_file")


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
