from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from object_tracking.logging import JsonlLogger
from object_tracking.unitree_g1 import (
    G1LocoSdk2Client,
    UnitreeG1Error,
    is_gstreamer_pipeline,
    g1_camera_candidates,
    parse_velocity,
)


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


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected bbox as 'x y width height'.")
    x, y, width, height = (int(float(part)) for part in parts)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Bounding box width and height must be positive.")
    return x, y, width, height


def open_camera(camera: int | str, robot_ip: str | None, robot_camera_url: str | None) -> tuple[cv2.VideoCapture, object, object]:
    sources: list[int | str]
    if camera == "g1":
        if robot_ip is None:
            raise RuntimeError("--camera g1 requires --robot-ip or --robot-camera-url.")
        sources = g1_camera_candidates(robot_ip, robot_camera_url)
    else:
        sources = [camera]

    errors: list[str] = []
    for source in sources:
        if isinstance(source, str) and is_gstreamer_pipeline(source):
            if not hasattr(cv2, "CAP_GSTREAMER"):
                raise RuntimeError(
                    "OpenCV in this environment does not expose CAP_GSTREAMER. Install a GStreamer-capable OpenCV build to use Unitree G1 camera streams."
                )
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{source}: could not open")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, frame, source
        cap.release()
        errors.append(f"{source}: opened but did not return a frame")

    details = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(f"Could not open camera source. Tried:\n{details}")


def build_g1_client(
    network_interface: str | None,
    robot_ip: str | None,
    needs_robot_commands: bool,
) -> G1LocoSdk2Client | None:
    if not needs_robot_commands:
        return None
    return G1LocoSdk2Client(
        network_interface=network_interface,
        robot_ip=robot_ip,
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
    robot_ip: str | None = None,
    robot_camera_url: str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    max_frames: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tracking.jsonl"
    video_path = output_dir / "tracking.mp4"

    cap, frame, camera_source = open_camera(camera, robot_ip, robot_camera_url)
    print(f"Using camera source: {camera_source}")

    if bbox is None:
        if not show:
            raise RuntimeError("Headless tracking requires --bbox 'x y width height'.")
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

                if max_frames is not None and frame_id >= max_frames:
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
        help="Camera index, video path, stream URL, or 'g1' to try common G1 stream URLs.",
    )
    parser.add_argument(
        "--robot-ip",
        help="Robot IP address. Used to resolve G1 camera URLs and SDK network interface.",
    )
    parser.add_argument(
        "--robot-camera-url",
        help="Explicit camera source. May include {ip}; examples: "
        'udpsrc address={ip} port=5600 ...',
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
        "--bbox",
        type=parse_bbox,
        metavar='"X Y WIDTH HEIGHT"',
        help="Initial tracking box. Required for headless tracking without ROI selection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many tracked frames.",
    )
    parser.add_argument(
        "--unitree-network-interface",
        help="Override the local network interface for SDK2 commands. Usually use --robot-ip instead.",
    )
    parser.add_argument(
        "--g1-command-on-start",
        choices=("none", "stand_up", "balance_stand", "stop_move", "damp"),
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
        needs_robot_commands = (
            args.g1_command_on_start != "none"
            or args.g1_velocity_on_start is not None
            or args.g1_stop_on_exit
        )
        g1_client = build_g1_client(
            args.unitree_network_interface,
            args.robot_ip,
            needs_robot_commands,
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
            robot_ip=args.robot_ip,
            robot_camera_url=args.robot_camera_url,
            bbox=args.bbox,
            max_frames=args.max_frames,
        )
    except UnitreeG1Error as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
