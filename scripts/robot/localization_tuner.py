#!/usr/bin/env python3
"""Interactive, no-actuation D435I localization sample collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="SPACE draws the plushie box; Q exits. Never commands the robot.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--serial", default="254322072511")
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("requires OpenCV and pyrealsense2 on the G1") from exc

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    device = profile.get_device()
    depth_scale = float(device.first_depth_sensor().get_depth_scale())
    window = "D435I tuner — SPACE select plushie, Q quit"
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(1000)
        while True:
            frames = align.process(pipeline.wait_for_frames(1000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            z16 = np.asanyarray(depth.get_data()).astype("<u2", copy=True)
            if rgb.shape[:2] != (540, 960) or z16.shape != (540, 960):
                raise RuntimeError(f"expected aligned 960x540 frames, got RGB={rgb.shape}, depth={z16.shape}")
            preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(preview, "SPACE: box plushie | Q: quit", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue
            x, y, width, height = (int(v) for v in cv2.selectROI(window, preview, False, True))
            if width <= 0 or height <= 0:
                continue
            entry = input("name torso_x torso_y torso_z metres: ").split()
            if len(entry) != 4:
                print("Skipped: expected name X Y Z")
                continue
            try:
                torso = [float(v) for v in entry[1:]]
                x2, y2 = x + width, y + height
                cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
                left, right = max(0, int(cx - width * 0.3)), min(960, int(cx + width * 0.3))
                top, bottom = max(0, int(cy - height * 0.3)), min(540, int(cy + height * 0.3))
                values = z16[top:bottom, left:right].astype(float) * depth_scale
                values = values[(values >= 0.12) & (values <= 4.0)]
                if len(values) < 8:
                    raise ValueError("not enough valid depth in selected box")
                dataset = {"samples": []}
                if args.dataset.exists():
                    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
                if any(sample.get("name") == entry[0] for sample in dataset.get("samples", [])):
                    raise ValueError(f"duplicate sample name {entry[0]!r}")
                frame_dir = args.dataset.parent / f"{args.dataset.stem}_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                np.save(frame_dir / f"{entry[0]}_rgb.npy", rgb)
                np.save(frame_dir / f"{entry[0]}_aligned_depth_z16.npy", z16)
                intrinsics = color.profile.as_video_stream_profile().get_intrinsics()
                dataset.update({
                    "camera_serial": device.get_info(rs.camera_info.serial_number),
                    "camera_firmware": device.get_info(rs.camera_info.firmware_version),
                    "rgb_profile": {"width": 960, "height": 540, "fps": 60, "format": "rgb8"},
                    "depth_profile": {"width": 960, "height": 540, "fps": 60, "format": "z16", "aligned_to_rgb": True},
                    "rgb_intrinsics": {"width": 960, "height": 540, "fx": float(intrinsics.fx), "fy": float(intrinsics.fy), "ppx": float(intrinsics.ppx), "ppy": float(intrinsics.ppy), "distortion_model": str(intrinsics.model).split(".")[-1].lower(), "coefficients": [float(v) for v in intrinsics.coeffs]},
                    "depth_scale": depth_scale,
                })
                dataset.setdefault("samples", []).append({
                    "name": entry[0], "bbox_xyxy": [x, y, x2, y2], "torso_truth_m": torso,
                    "median_aligned_depth_m": float(np.median(values)), "depth_valid_fraction": float(len(values) / ((right-left)*(bottom-top))),
                    "rgb_file": str(frame_dir / f"{entry[0]}_rgb.npy"), "aligned_depth_file": str(frame_dir / f"{entry[0]}_aligned_depth_z16.npy"), "captured_unix_s": time.time(),
                })
                args.dataset.parent.mkdir(parents=True, exist_ok=True)
                args.dataset.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
                print(f"Saved {entry[0]}: depth={np.median(values):.3f} m, bbox={x,y,x2,y2}")
            except ValueError as exc:
                print(f"Skipped: {exc}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
