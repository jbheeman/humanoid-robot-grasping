from __future__ import annotations

import argparse
from pathlib import Path

from object_tracking.manual_tracker import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track an object from a G1 camera stream.")
    parser.add_argument("robot_ip", help="Robot IP address.")
    parser.add_argument(
        "--camera-url",
        help="Explicit camera URL. May include {ip}; otherwise common G1 RTSP-style URLs are tried.",
    )
    parser.add_argument(
        "--output",
        default="runs/g1_vision_test",
        help="Directory for tracking logs and video.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable display window.")
    parser.add_argument("--no-record", action="store_true", help="Disable annotated MP4 recording.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        camera="g1",
        output_dir=Path(args.output),
        show=not args.no_show,
        record=not args.no_record,
        robot_ip=args.robot_ip,
        robot_camera_url=args.camera_url,
    )
