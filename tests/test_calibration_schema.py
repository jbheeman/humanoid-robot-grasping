from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np

from object_tracking.arm_tracking.calibration import (
    Calibration,
    CalibrationError,
    CalibrationPose,
    RegistrationResiduals,
    ResidualSummary,
    StreamProfile,
    load_calibration,
    save_calibration_atomic,
    solve_rigid_transform,
    tune_transform,
    validate_correspondences,
    validate_registration_correspondences,
)
from object_tracking.arm_tracking.calibration_cli import (
    append_correspondence,
    main as calibration_main,
    retune_calibration,
    solve_from_files,
)
from object_tracking.arm_tracking.geometry import CameraIntrinsics, RigidTransform, WorkspaceBounds


def make_calibration() -> Calibration:
    profile = StreamProfile(640, 480, 30, "z16")
    rgb_profile = StreamProfile(640, 480, 30, "rgb8")
    intrinsics = CameraIntrinsics(640, 480, 500, 500, 320, 240)
    solve_poses = tuple(
        CalibrationPose(
            f"solve-{index}",
            (index * 0.01, index % 2, index % 3),
            (index * 0.01, index % 2, index % 3),
        )
        for index in range(8)
    )
    validation_poses = tuple(
        CalibrationPose(
            f"validate-{index}", (index * 0.02, index % 2, 0.2), (index * 0.02, index % 2, 0.2)
        )
        for index in range(4)
    )
    calibration = Calibration(
        calibration_id=str(uuid.uuid4()),
        calibration_hash="",
        created_at="2026-07-09T12:00:00+00:00",
        camera_serial="camera-123",
        camera_firmware="5.16.0",
        rgb_profile=rgb_profile,
        depth_profile=profile,
        rgb_intrinsics=intrinsics,
        depth_intrinsics=intrinsics,
        depth_scale=0.001,
        depth_to_rgb=RigidTransform.identity(),
        optical_to_torso=RigidTransform.identity(),
        waist_reference_rad=(0.0, 0.0, 0.0),
        tag_to_wrist=RigidTransform.identity(),
        workspace=WorkspaceBounds([0.1, -0.5, 0.1], [0.8, 0.5, 1.0]),
        solve_poses=solve_poses,
        validation_poses=validation_poses,
        solve_residuals=ResidualSummary(0.005, 0.010, 1.0),
        validation_residuals=ResidualSummary(0.010, 0.020, 2.0),
        registration_residuals=RegistrationResiduals(1.0, 2.0, 8),
    )
    return calibration.with_computed_hash()


class CalibrationSolveTests(unittest.TestCase):
    def test_solves_known_rigid_transform(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        expected = RigidTransform.from_xyz_rpy([0.2, -0.1, 0.3], [0.1, -0.2, 0.3])
        target = expected.apply(source)

        result = solve_rigid_transform(source, target)

        np.testing.assert_allclose(result.transform.rotation, expected.rotation, atol=1e-10)
        np.testing.assert_allclose(result.transform.translation, expected.translation, atol=1e-10)
        self.assertLess(result.residuals.rmse_m, 1e-10)

    def test_held_out_validation_calculates_metric_errors(self) -> None:
        poses = [
            CalibrationPose("a", (0, 0, 0), (0.01, 0, 0), orientation_error_deg=2.0),
            CalibrationPose("b", (1, 0, 0), (1, 0, 0), orientation_error_deg=1.0),
        ]
        summary = validate_correspondences(RigidTransform.identity(), poses)
        self.assertAlmostEqual(summary.rmse_m, np.sqrt(0.0001 / 2))
        self.assertEqual(summary.max_orientation_error_deg, 2.0)

    def test_tuning_is_bounded(self) -> None:
        tuned = tune_transform(
            RigidTransform.identity(), xyz_delta_m=[0.01, 0, 0], rpy_delta_deg=[0, 0, 1]
        )
        np.testing.assert_allclose(tuned.translation, [0.01, 0, 0])
        with self.assertRaisesRegex(CalibrationError, "translation"):
            tune_transform(RigidTransform.identity(), xyz_delta_m=[0.06, 0, 0])

    def test_registration_probe_reports_pixel_error(self) -> None:
        expected = [(0, 0), (10, 0), (0, 10), (10, 10)]
        measured = [(1, 0), (11, 0), (0, 12), (10, 10)]
        residuals = validate_registration_correspondences(expected, measured)
        self.assertEqual(residuals.median_error_px, 1.0)
        self.assertEqual(residuals.max_error_px, 2.0)


class CalibrationSchemaTests(unittest.TestCase):
    def test_round_trips_dict_and_detects_tampering(self) -> None:
        calibration = make_calibration()
        round_tripped = Calibration.from_dict(calibration.to_dict())
        self.assertEqual(round_tripped.calibration_hash, calibration.calibration_hash)

        tampered = calibration.to_dict()
        tampered["camera"]["serial"] = "other-camera"
        with self.assertRaisesRegex(CalibrationError, "hash"):
            Calibration.from_dict(tampered)

    def test_execution_requires_matching_camera_profiles_waist_and_residuals(self) -> None:
        calibration = make_calibration()
        calibration.validate_for_execution(
            camera_serial="camera-123",
            rgb_profile=calibration.rgb_profile,
            depth_profile=calibration.depth_profile,
            waist_rad=[0, 0, np.deg2rad(2.9)],
            calibration_id=calibration.calibration_id,
        )
        with self.assertRaisesRegex(CalibrationError, "waist deviates"):
            calibration.validate_for_execution(
                camera_serial="camera-123",
                rgb_profile=calibration.rgb_profile,
                depth_profile=calibration.depth_profile,
                waist_rad=[0, 0, np.deg2rad(3.1)],
            )

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is optional")
    def test_yaml_save_is_atomic_and_loads_validated_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            saved = save_calibration_atomic(make_calibration(), path)
            loaded = load_calibration(path)
            self.assertEqual(loaded.calibration_hash, saved.calibration_hash)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is optional")
class CalibrationCliTests(unittest.TestCase):
    def test_collect_solve_validate_and_tune_workflow(self) -> None:
        import yaml

        source = make_calibration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            collection = root / "collection.yaml"
            solved_path = root / "solved.yaml"
            tuned_path = root / "tuned.yaml"
            template.write_text(yaml.safe_dump(source.to_dict(), sort_keys=False))
            for subset, poses in (
                ("solve", source.solve_poses),
                ("validation", source.validation_poses),
            ):
                for pose in poses:
                    append_correspondence(
                        collection,
                        subset=subset,
                        name=pose.name,
                        optical_point_m=pose.optical_point_m,
                        torso_point_m=pose.torso_point_m,
                        source="apriltag",
                    )

            solved = solve_from_files(template, collection, solved_path)

            self.assertLess(solved.validation_residuals.rmse_m, 1e-10)
            self.assertEqual(calibration_main(["validate", str(solved_path)]), 0)
            tuned = retune_calibration(
                solved_path,
                tuned_path,
                xyz_delta_m=(0.001, 0, 0),
                rpy_delta_deg=(0, 0, 0),
            )
            self.assertAlmostEqual(tuned.validation_residuals.rmse_m, 0.001)

    def test_collect_rejects_duplicate_pose_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = {
                "path": Path(directory) / "collection.yaml",
                "subset": "solve",
                "name": "pose-a",
                "optical_point_m": (0, 0, 1),
                "torso_point_m": (0, 0, 1),
                "source": "manual",
            }
            append_correspondence(**values)
            with self.assertRaisesRegex(CalibrationError, "already exists"):
                append_correspondence(**values)


if __name__ == "__main__":
    unittest.main()
