from dataclasses import replace

import numpy as np

from object_tracking.arm_tracking.calibration import (
    Calibration,
    CalibrationPose,
    ResidualSummary,
    StreamProfile,
)
from object_tracking.arm_tracking.geometry import CameraIntrinsics, RigidTransform, WorkspaceBounds
from object_tracking.arm_tracking.runtime import register_depth_in_rgb


def _calibration() -> Calibration:
    profile = StreamProfile(4, 3, 30, "z16")
    intrinsics = CameraIntrinsics(4, 3, 2.0, 2.0, 1.5, 1.0)
    pose = CalibrationPose("p", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0, 0.0)
    residuals = ResidualSummary(0.0, 0.0, 0.0)
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
        solve_poses=(pose,) * 8,
        validation_poses=(pose,) * 4,
        solve_residuals=residuals,
        validation_residuals=residuals,
    )
    return calibration.with_computed_hash()


def test_identity_registration_preserves_valid_depth_pixels() -> None:
    z16 = np.zeros((3, 4), dtype=np.uint16)
    z16[1, 2] = 1000
    aligned = register_depth_in_rgb(z16, _calibration())
    assert aligned.shape == z16.shape
    assert aligned[1, 2] == 1000
