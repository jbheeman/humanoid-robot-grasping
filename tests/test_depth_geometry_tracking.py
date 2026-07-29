from __future__ import annotations

import unittest

import numpy as np

from object_tracking.arm_tracking.depth import (
    DepthFrame,
    DepthFrameBuffer,
    estimate_adaptive_roi_depth,
    estimate_roi_depth,
    pair_rgb_depth,
)
from object_tracking.arm_tracking.geometry import (
    CameraIntrinsics,
    Plane,
    RigidTransform,
    SupportRegion,
    WorkspaceBounds,
    detect_automatic_support_region,
    deproject_pixel,
    extract_support_plane,
    fit_plane,
    generate_pregrasp_target,
    has_plane_clearance,
    has_support_clearance,
    intersect_pixel_ray_with_plane,
    map_pixel_between_profiles,
)
from object_tracking.arm_tracking.tracking import (
    PositionVelocityFilter,
    StickyTargetSelector,
    TargetCandidate,
)


def frame(sequence: int, receipt: float) -> DepthFrame:
    return DepthFrame(
        sequence=sequence,
        receipt_time_s=receipt,
        z16=np.zeros((2, 2), dtype=np.uint16),
        depth_scale=0.001,
        calibration_id="calibration-a",
    )


class DepthPairingTests(unittest.TestCase):
    def test_pairs_nearest_by_receipt_time(self) -> None:
        pair = pair_rgb_depth(10.0, [frame(1, 9.93), frame(2, 10.02)])
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.depth.sequence, 2)
        self.assertAlmostEqual(pair.skew_s, 0.02)

    def test_rejects_pair_over_100ms(self) -> None:
        self.assertIsNone(pair_rgb_depth(10.0, [frame(1, 9.899)]))

    def test_buffer_rejects_replayed_sequence(self) -> None:
        buffer = DepthFrameBuffer()
        buffer.add(frame(2, 1.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            buffer.add(frame(2, 1.1))


class RobustDepthTests(unittest.TestCase):
    def test_selects_supported_foreground_cluster(self) -> None:
        depth = np.full((20, 20), 2000, dtype=np.uint16)
        depth[6:14, 6:14] = 1000
        depth[8, 8] = 400
        depth[9, 9] = 0

        estimate = estimate_roi_depth(depth, [3, 3, 17, 17], depth_scale=0.001)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.depth_m, 1.0)
        self.assertGreater(estimate.sample_count, 20)
        self.assertTrue(estimate.is_certain)

    def test_returns_none_for_sparse_valid_depth(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        depth[10, 10] = 1000
        self.assertIsNone(estimate_roi_depth(depth, [2, 2, 18, 18], depth_scale=0.001))

    def test_adaptive_roi_uses_coherent_object_center(self) -> None:
        gradient = np.linspace(600, 1000, 80, dtype=np.uint16)
        depth = np.repeat(gradient[np.newaxis, :], 80, axis=0)
        depth[36:44, 36:44] = 800

        large = estimate_roi_depth(
            depth,
            [0, 0, 80, 80],
            depth_scale=0.001,
            roi_fraction=0.6,
        )
        estimate = estimate_adaptive_roi_depth(
            depth,
            [0, 0, 80, 80],
            depth_scale=0.001,
        )

        self.assertIsNotNone(large)
        assert large is not None
        self.assertFalse(large.is_certain)
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertTrue(estimate.is_certain)
        self.assertAlmostEqual(estimate.depth_m, 0.8, places=3)

    def test_adaptive_roi_recovers_from_fluffy_object_center_hole(self) -> None:
        depth = np.full((100, 100), 1100, dtype=np.uint16)
        depth[20:80, 20:80] = 0
        depth[28:45, 28:45] = 720

        estimate = estimate_adaptive_roi_depth(
            depth,
            [20, 20, 80, 80],
            depth_scale=0.001,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertTrue(estimate.is_certain)
        self.assertAlmostEqual(estimate.depth_m, 0.72, places=3)
        self.assertGreaterEqual(estimate.sample_count, 8)
        self.assertGreaterEqual(estimate.pixel_xy[0], 28)
        self.assertLess(estimate.pixel_xy[0], 45)

    def test_adaptive_roi_fallback_stays_inside_detection(self) -> None:
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[10:20, 10:20] = 600
        depth[80:90, 80:90] = 600

        self.assertIsNone(
            estimate_adaptive_roi_depth(
                depth,
                [20, 20, 80, 80],
                depth_scale=0.001,
            )
        )

    def test_adaptive_roi_rejects_fallback_patch_outside_detection(self) -> None:
        depth = np.ones((20, 20), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "inside the detection box"):
            estimate_adaptive_roi_depth(
                depth,
                [0, 0, 20, 20],
                depth_scale=0.001,
                fallback_offsets=(-0.5, 0.0, 0.5),
            )


class GeometryTests(unittest.TestCase):
    def test_deprojects_and_transforms(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500, 500, 320, 240)
        optical = deproject_pixel((370, 190), 2.0, intrinsics)
        np.testing.assert_allclose(optical, [0.2, -0.2, 2.0])

        transform = RigidTransform.from_xyz_rpy([1, 2, 3], [0, 0, np.pi / 2])
        torso = transform.apply(optical)
        np.testing.assert_allclose(torso, [1.2, 2.2, 5.0], atol=1e-9)
        np.testing.assert_allclose(transform.inverse().apply(torso), optical, atol=1e-9)

    def test_intersects_calibrated_pixel_ray_with_plane(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500, 500, 320, 240)
        point = intersect_pixel_ray_with_plane(
            (370, 190),
            intrinsics,
            RigidTransform.identity(),
            Plane((0, 0, 1), -2.0),
        )
        np.testing.assert_allclose(point, (0.2, -0.2, 2.0), atol=1e-9)

    def test_deprojects_zero_coefficient_brown_profile(self) -> None:
        intrinsics = CameraIntrinsics(
            640,
            480,
            500,
            500,
            320,
            240,
            distortion_model="brown_conrady",
            coefficients=(0, 0, 0, 0, 0),
        )
        np.testing.assert_allclose(deproject_pixel((370, 190), 2.0, intrinsics), [0.2, -0.2, 2.0])

    def test_maps_resized_detection_back_to_native_profile(self) -> None:
        self.assertEqual(
            map_pixel_between_profiles((320, 320), (640, 640), (1280, 720)), (640, 360)
        )

    def test_plane_clearance_workspace_and_pregrasp(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        plane = fit_plane(points, orient_toward=[0, 0, 1])
        self.assertTrue(has_plane_clearance([0.2, 0.2, 0.11], plane))
        self.assertFalse(has_plane_clearance([0.2, 0.2, 0.09], plane))

        target = generate_pregrasp_target([0.6, 0, 0.5], [0, 0, 0.5])
        np.testing.assert_allclose(target.position, [0.4, 0, 0.5])
        workspace = WorkspaceBounds([0.1, -0.5, 0.1], [0.8, 0.5, 1.0])
        self.assertTrue(workspace.contains(target.position))

    def test_support_region_projects_only_over_certified_table_footprint(self) -> None:
        support = SupportRegion.from_xy_bounds(
            Plane((0.0, 0.0, 1.0), 0.0),
            (0.0, -0.5),
            (1.0, 0.5),
            certified_edges=("u_min", "u_max", "v_min", "v_max"),
            lateral_margin_m=0.05,
        )
        np.testing.assert_allclose(
            support.project_to_clearance(
                (0.5, 0.0, -0.02),
                minimum_clearance_m=0.05,
            ),
            (0.5, 0.0, 0.05),
        )
        np.testing.assert_allclose(
            support.project_to_clearance(
                (-0.2, 0.0, -0.02),
                minimum_clearance_m=0.05,
            ),
            (-0.2, 0.0, -0.02),
        )

    def test_extracts_dominant_plane_from_hardware_depth(self) -> None:
        intrinsics = CameraIntrinsics(32, 24, 30, 30, 16, 12)
        depth = np.full((24, 32), 1000, dtype=np.uint16)
        depth[16:20, 12:16] = 1500
        plane, inliers = extract_support_plane(
            depth,
            intrinsics,
            depth_scale=0.001,
            lower_image_fraction=0.5,
            stride=1,
        )
        self.assertGreater(float(inliers.mean()), 0.9)
        self.assertAlmostEqual(abs(plane.offset), 1.0, places=3)

    def test_support_plane_bounds_exclude_dominant_background(self) -> None:
        intrinsics = CameraIntrinsics(32, 24, 30, 30, 16, 12)
        depth = np.full((24, 32), 1800, dtype=np.uint16)
        depth[8:22, 6:27] = 700
        optical_to_base = RigidTransform(
            np.eye(3),
            (0.5, 0.0, -0.67),
        )
        plane, _ = extract_support_plane(
            depth,
            intrinsics,
            depth_scale=0.001,
            optical_to_base=optical_to_base,
            pixel_roi=((5, 7), (28, 7), (28, 23), (5, 23)),
            base_minimum=(0.0, -1.0, -0.05),
            base_maximum=(1.0, 1.0, 0.10),
            stride=1,
        )
        self.assertAlmostEqual(float(plane.offset), -0.03, places=3)

    def test_bounded_support_region_releases_only_across_certified_edge(self) -> None:
        support = SupportRegion.from_xy_bounds(
            Plane((0.0, 0.0, 1.0), 0.0),
            (0.37, -0.35),
            (0.80, 0.35),
            certified_edges=("u_min",),
            lateral_margin_m=0.07,
        )

        self.assertTrue(
            has_support_clearance((0.20, 0.0, -0.10), support, minimum_clearance_m=0.05)
        )
        self.assertFalse(
            has_support_clearance((0.31, 0.0, -0.10), support, minimum_clearance_m=0.05)
        )
        self.assertFalse(
            has_support_clearance((0.50, 0.80, -0.10), support, minimum_clearance_m=0.05)
        )
        self.assertTrue(has_support_clearance((0.50, 0.0, 0.05), support, minimum_clearance_m=0.05))

    def test_support_region_round_trips_without_losing_certification(self) -> None:
        support = SupportRegion.from_xy_bounds(
            Plane((0.02, -0.01, 0.99975), 0.005),
            (0.37, -0.35),
            (0.80, 0.35),
            certified_edges=("u_min",),
        )
        restored = SupportRegion.from_dict(support.to_dict())

        np.testing.assert_allclose(restored.plane.normal, support.plane.normal)
        np.testing.assert_allclose(restored.minimum_uv, support.minimum_uv)
        self.assertEqual(restored.certified_edges, ("u_min",))

    def test_support_region_uses_explicit_ordered_tabletop_corners(self) -> None:
        plane = Plane((0.0, 0.0, 1.0), 0.0)
        support = SupportRegion.from_ordered_corners(
            plane,
            (
                (0.40, 0.30, 0.0),
                (0.40, -0.30, 0.0),
                (0.75, -0.30, 0.0),
                (0.75, 0.30, 0.0),
            ),
        )

        self.assertEqual(
            set(support.certified_edges),
            {"u_min", "u_max", "v_min", "v_max"},
        )
        self.assertTrue(support.has_clearance((0.20, 0.0, -0.10), minimum_clearance_m=0.05))
        self.assertFalse(support.has_clearance((0.55, 0.0, -0.10), minimum_clearance_m=0.05))

    def test_pixel_corner_region_uses_physical_dimension_prior(self) -> None:
        support = SupportRegion.from_ordered_corners(
            Plane((0.0, 0.0, 1.0), 0.0),
            (
                (0.40, 0.385, 0.0),
                (0.40, -0.385, 0.0),
                (0.865, -0.385, 0.0),
                (0.865, 0.385, 0.0),
            ),
        ).with_dimension_prior(
            (0.34, 0.68),
            source="live_plane_calibrated_near_edge_dimension_prior",
        )

        np.testing.assert_allclose(
            support.maximum_uv - support.minimum_uv,
            (0.34, 0.68),
        )
        self.assertEqual(support.certified_edges, ("u_min",))
        self.assertEqual(support.edge_source("u_min"), "calibrated_pixel_near_edge")
        self.assertEqual(support.edge_source("u_max"), "dimension_prior")
        np.testing.assert_allclose(support.axis_u, (1.0, 0.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(support.axis_v, (0.0, -1.0, 0.0), atol=1e-9)

    def test_support_region_rejects_non_table_plane(self) -> None:
        with self.assertRaisesRegex(ValueError, "tabletop-like"):
            SupportRegion.from_xy_bounds(
                Plane((1.0, 0.0, 0.0), 0.0),
                (0.37, -0.35),
                (0.80, 0.35),
            )

    def test_automatic_support_uses_live_plane_and_dimension_prior(self) -> None:
        intrinsics = CameraIntrinsics(120, 100, 100.0, 100.0, 60.0, 50.0)
        depth = np.zeros((100, 120), dtype=np.uint16)
        depth[20:92, 18:108] = 1000

        support, diagnostics = detect_automatic_support_region(
            depth,
            intrinsics,
            depth_scale=0.001,
            optical_to_base=RigidTransform.identity(),
            base_minimum=(-0.6, -0.6, 0.9),
            base_maximum=(0.6, 0.6, 1.1),
            expected_size_m=(0.60, 0.70),
            stride=6,
            minimum_connected_inliers=60,
        )

        self.assertEqual(support.source, "automatic_rgbd_plane_dimension_prior")
        self.assertEqual(support.certified_edges, ("u_min",))
        self.assertEqual(support.edge_source("u_max"), "prior_estimated")
        np.testing.assert_allclose(
            support.maximum_uv - support.minimum_uv,
            (0.60, 0.70),
            atol=1e-9,
        )
        self.assertLess(diagnostics.normal_tilt_deg, 0.1)
        self.assertGreater(diagnostics.connected_inlier_count, 60)

    def test_support_region_classifies_inflated_table_prism(self) -> None:
        support = SupportRegion.from_xy_bounds(
            Plane((0.0, 0.0, 1.0), 0.0),
            (0.4, -0.3),
            (0.8, 0.3),
        )

        self.assertEqual(
            support.classify_point((0.5, 0.0, 0.01), side_margin_m=0.07),
            "under_or_inside",
        )
        self.assertEqual(
            support.classify_point((0.2, 0.0, 0.01), side_margin_m=0.07),
            "outside",
        )
        self.assertEqual(
            support.classify_point((0.5, 0.0, 0.10), side_margin_m=0.07),
            "above_clearance",
        )


class TrackingTests(unittest.TestCase):
    def test_filter_estimates_velocity_and_predicts(self) -> None:
        tracker = PositionVelocityFilter(
            position_gain=1.0,
            velocity_gain=1.0,
            velocity_damping=1.0,
            max_speed_mps=2.0,
            max_acceleration_mps2=20.0,
            max_innovation_m=1.0,
            max_prediction_displacement_m=1.0,
        )
        tracker.update([0, 0, 1], 1.0)
        tracker.update([0.1, 0, 1], 1.1)
        prediction = tracker.predict(0.15)
        assert prediction is not None
        np.testing.assert_allclose(prediction, [0.25, 0, 1], atol=1e-9)

    def test_filter_bounds_single_frame_jitter_and_prediction(self) -> None:
        tracker = PositionVelocityFilter()
        tracker.update([0.5, 0.0, 0.0], 1.0)
        state = tracker.update([0.7, 0.0, 0.0], 1.02)
        prediction = tracker.predict(0.15)
        assert prediction is not None
        self.assertLessEqual(state.position_m[0] - 0.5, 0.014)
        self.assertLessEqual(np.linalg.norm(state.velocity_mps), 0.041)
        self.assertLessEqual(np.linalg.norm(prediction - state.position_m), 0.06)
        self.assertAlmostEqual(tracker.last_residual_m, 0.2)

    def test_filter_damps_alternating_measurement_jitter(self) -> None:
        tracker = PositionVelocityFilter()
        tracker.update([0.5, 0.0, 0.0], 1.0)
        predictions = []
        for index in range(1, 21):
            jitter = 0.02 if index % 2 else -0.02
            tracker.update([0.5 + jitter, 0.0, 0.0], 1.0 + index * 0.02)
            prediction = tracker.predict(0.15)
            assert prediction is not None
            predictions.append(prediction[0])
        self.assertLess(max(predictions[-10:]) - min(predictions[-10:]), 0.025)

    def test_filter_resets_after_gap(self) -> None:
        tracker = PositionVelocityFilter(reset_gap_s=0.2)
        tracker.update([0, 0, 0], 1.0)
        state = tracker.update([1, 0, 0], 1.3)
        np.testing.assert_array_equal(state.velocity_mps, np.zeros(3))

    def test_selector_holds_identity_then_releases_after_loss(self) -> None:
        selector = StickyTargetSelector(loss_timeout_s=0.2)
        first = TargetCandidate(1, 0.8, [0, 0, 1])
        other = TargetCandidate(2, 0.9, [0, 0, 1])
        self.assertEqual(selector.update([first, other], 1.0).track_id, 2)
        self.assertIsNone(selector.update([first], 1.1))
        self.assertEqual(selector.update([first], 1.21).track_id, 1)


if __name__ == "__main__":
    unittest.main()
