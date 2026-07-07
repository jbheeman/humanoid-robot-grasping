from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from object_tracking.logging import JsonlLogger
from object_tracking.unitree_g1 import G1LocoSdk2Client, UnitreeG1Error, parse_velocity


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


def build_g1_client(
    network_interface: str | None,
    sdk2_path: Path,
    loco_binary: Path | None,
) -> G1LocoSdk2Client | None:
    if not network_interface:
        return None
    return G1LocoSdk2Client(
        network_interface=network_interface,
        sdk2_path=sdk2_path,
        loco_binary=loco_binary,
    )


def log_g1_result(logger: JsonlLogger, event: str, result: object) -> None:
    if result is None:
        return
    logger.write(
        {
            "event": event,
            "timestamp": time.time(),
            "command": getattr(result, "command", []),
            "returncode": getattr(result, "returncode", None),
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
        }
    )


def run(
    camera: int | str,
    output_dir: Path,
    show: bool,
    record: bool,
    g1_client: G1LocoSdk2Client | None = None,
    g1_command_on_start: str = "none",
    g1_velocity_on_start: tuple[float, float, float, float | None] | None = None,
    g1_stop_on_exit: bool = False,
) -> None:
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
        if g1_client is not None:
            if g1_command_on_start != "none":
                log_g1_result(
                    logger,
                    "g1_command_on_start",
                    g1_client.command(g1_command_on_start),
                )
            if g1_velocity_on_start is not None:
                log_g1_result(
                    logger,
                    "g1_velocity_on_start",
                    g1_client.move(*g1_velocity_on_start),
                )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = time.time()
                frame_id += 1
                track_ok, tracked_bbox = tracker.update(frame)

                x, y, w, h = [float(v) for v in tracked_bbox]
                tracking_record = {
                    "event": "tracking_frame",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "elapsed_s": timestamp - start_time,
                    "track_ok": bool(track_ok),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                }
                logger.write(tracking_record)

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
        finally:
            if g1_client is not None and g1_stop_on_exit:
                log_g1_result(logger, "g1_stop_on_exit", g1_client.stop_move())

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
    parser.add_argument(
        "--unitree-network-interface",
        help="Network interface connected to the G1. Enables SDK2 G1 loco commands.",
    )
    parser.add_argument(
        "--unitree-sdk2-path",
        type=Path,
        default=Path("~/Documents/unitree_sdk2").expanduser(),
        help="Path to the Unitree SDK2 checkout.",
    )
    parser.add_argument(
        "--unitree-loco-binary",
        type=Path,
        help="Path to a built g1_loco_client binary. Defaults to SDK2 build outputs.",
    )
    parser.add_argument(
        "--g1-command-on-start",
        choices=("none", "get_fsm_id", "start", "stand_up", "balance_stand", "stop_move", "damp"),
        default="none",
        help="Optional one-shot G1 loco command after tracker initialization.",
    )
    parser.add_argument(
        "--g1-velocity-on-start",
        type=parse_velocity,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Optional one-shot G1 velocity command after tracker initialization.",
    )
    parser.add_argument(
        "--g1-stop-on-exit",
        action="store_true",
        help="Send G1 stop_move when the tracker exits.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        g1_client = build_g1_client(
            args.unitree_network_interface,
            args.unitree_sdk2_path,
            args.unitree_loco_binary,
        )
        run(
            parse_camera(args.camera),
            Path(args.output),
            show=not args.no_show,
            record=not args.no_record,
            g1_client=g1_client,
            g1_command_on_start=args.g1_command_on_start,
            g1_velocity_on_start=args.g1_velocity_on_start,
            g1_stop_on_exit=args.g1_stop_on_exit,
        )
    except UnitreeG1Error as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
