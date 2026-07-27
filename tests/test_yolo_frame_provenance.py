import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")

from object_tracking.simple_tracker import SimpleTracker
from object_tracking.yolo_stream_server import (
    ProcessedFrameBundle,
    arm_tracking_snapshot,
    current_detection_tracks,
    processed_frame_is_newer,
    state,
)


def detection(center_x: float) -> dict[str, object]:
    return {
        "class_name": "bunny",
        "confidence": 0.9,
        "bbox_xyxy": [center_x - 1.0, 0.0, center_x + 1.0, 2.0],
        "center_xy": [center_x, 1.0],
    }


def test_retained_unmatched_track_keeps_last_observation_time() -> None:
    tracker = SimpleTracker(max_missed_updates=2)
    first = tracker.update([detection(10.0)], 5.0)
    retained = tracker.update([], 6.0)

    assert first[0]["last_seen"] == 5.0
    assert retained[0]["last_seen"] == 5.0
    assert retained[0]["missed_updates"] == 1
    assert current_detection_tracks(retained) == ()


def test_duplicate_or_out_of_order_processed_frame_is_rejected() -> None:
    def bundle(frame_id: int) -> ProcessedFrameBundle:
        return ProcessedFrameBundle(
            frame_id=frame_id,
            frame_receipt_monotonic_s=float(frame_id),
            frame_shape=(3, 4, 3),
            inference_started_monotonic_s=float(frame_id) + 0.01,
            inference_completed_monotonic_s=float(frame_id) + 0.02,
            detections=(),
            tracks=(),
        )

    existing = bundle(7)

    assert not processed_frame_is_newer(existing, bundle(7))
    assert not processed_frame_is_newer(existing, bundle(6))
    assert processed_frame_is_newer(existing, bundle(8))


def test_arm_snapshot_uses_atomic_processed_bundle_not_newest_raw_frame() -> None:
    bundle = ProcessedFrameBundle(
        frame_id=7,
        frame_receipt_monotonic_s=10.0,
        frame_shape=(540, 960, 3),
        inference_started_monotonic_s=10.01,
        inference_completed_monotonic_s=10.04,
        detections=(),
        tracks=(
            {
                "track_id": 1,
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                "missed_updates": 0,
            },
        ),
    )
    with state.lock:
        previous = state.processed_frame
        previous_raw = state.raw_frame
        previous_raw_time = state.raw_frame_received_monotonic
        previous_count = state.frame_count
        state.processed_frame = bundle
        state.raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state.raw_frame_received_monotonic = 11.0
        state.frame_count = 12
    try:
        snapshot = arm_tracking_snapshot()
    finally:
        with state.lock:
            state.processed_frame = previous
            state.raw_frame = previous_raw
            state.raw_frame_received_monotonic = previous_raw_time
            state.frame_count = previous_count

    assert snapshot["rgb_frame_id"] == 7
    assert snapshot["rgb_receipt_time_s"] == 10.0
    assert snapshot["rgb_shape"] == (540, 960, 3)
    assert snapshot["tracks"][0]["track_id"] == 1
