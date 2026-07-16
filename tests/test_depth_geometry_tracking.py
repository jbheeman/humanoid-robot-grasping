from __future__ import annotations

import unittest

import numpy as np

from object_tracking.arm_tracking.depth import (
    DepthFrame,
    DepthFrameBuffer,
    estimate_roi_depth,
    pair_rgb_depth,
)
from object_tracking.arm_tracking.geometry import (
    CameraIntrinsics,
    Plane,
    RigidTransform,
    SupportRegion,
    WorkspaceBounds,
    deproject_pixel,
    extract_support_plane,
    fit_plane,
    generate_pregrasp_target,
    has_plane_clearance,
    has_support_clearance,
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


class GeometryTests(unittest.TestCase):
    def test_deprojects_and_transforms(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500, 500, 320, 240)
        optical = deproject_pixel((370, 190), 2.0, intrinsics)
        np.testing.assert_allclose(optical, [0.2, -0.2, 2.0])

        transform = RigidTransform.from_xyz_rpy([1, 2, 3], [0, 0, np.pi / 2])
        torso = transform.apply(optical)
        np.testing.assert_allclose(torso, [1.2, 2.2, 5.0], atol=1e-9)
        np.testing.assert_allclose(transform.inverse().apply(torso), optical, atol=1e-9)

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

    def test_support_region_rejects_non_table_plane(self) -> None:
        with self.assertRaisesRegex(ValueError, "tabletop-like"):
            SupportRegion.from_xy_bounds(
                Plane((1.0, 0.0, 0.0), 0.0),
                (0.37, -0.35),
                (0.80, 0.35),
            )


class TrackingTests(unittest.TestCase):
    def test_filter_estimates_velocity_and_predicts(self) -> None:
        tracker = PositionVelocityFilter(position_gain=1.0, velocity_gain=1.0)
        tracker.update([0, 0, 1], 1.0)
        tracker.update([0.1, 0, 1], 1.1)
        prediction = tracker.predict(0.15)
        assert prediction is not None
        np.testing.assert_allclose(prediction, [0.25, 0, 1], atol=1e-9)

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
