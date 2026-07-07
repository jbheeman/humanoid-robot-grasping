from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from object_tracking.logging import JsonlLogger


def create_tracker() -> cv2.Tracker:
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError(
        "No OpenCV CSRT/KCF tracker available. Try installing opencv-contrib-python."
    )


def parse_camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def run(camera: int | str, output_dir: Path, show: bool, record: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tracking.jsonl"
    video_path = output_dir / "tracking.mp4"

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {camera}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from camera/video source")

    bbox = cv2.selectROI("Select plush object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select plush object")
    if bbox == (0, 0, 0, 0):
        raise RuntimeError("No bounding box selected")

    tracker = create_tracker()
    tracker.init(frame, bbox)

    frame_id = 0
    start_time = time.time()
    video_writer = None
    if record:
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not video_writer.isOpened():
            video_writer.release()
            video_writer = None
            print(f"Could not open video writer for {video_path}; continuing without video.")

    with JsonlLogger(log_path) as logger:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = time.time()
            frame_id += 1
            track_ok, tracked_bbox = tracker.update(frame)

            x, y, w, h = [float(v) for v in tracked_bbox]
            record = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "elapsed_s": timestamp - start_time,
                "track_ok": bool(track_ok),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
            }
            logger.write(record)

            if track_ok:
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "object",
                    (int(x), max(0, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "track lost",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            if video_writer is not None:
                video_writer.write(frame)

            if show:
                cv2.imshow("Object tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"Wrote tracking log to {log_path}")
    if record and video_writer is not None:
        print(f"Wrote annotated video to {video_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual bbox tracker for a plush object.")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or video path. Use 0 for default webcam.",
    )
    parser.add_argument(
        "--output",
        default="runs/manual_tracker",
        help="Directory for tracking logs.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable display window.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable annotated MP4 recording.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        parse_camera(args.camera),
        Path(args.output),
        show=not args.no_show,
        record=not args.no_record,
    )


if __name__ == "__main__":
    main()
