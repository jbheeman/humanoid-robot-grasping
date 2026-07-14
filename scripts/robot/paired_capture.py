#!/usr/bin/env python3
"""Capture one locally aligned D435I RGB/depth sample; never actuates the G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


def _intrinsics(value: Any) -> dict[str, Any]:
    return {
        "width": int(value.width), "height": int(value.height), "fx": float(value.fx), "fy": float(value.fy),
        "ppx": float(value.ppx), "ppy": float(value.ppy), "distortion_model": str(value.model).split(".")[-1].lower(),
        "coefficients": [float(item) for item in value.coeffs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture local aligned D435I RGB/depth; no network or arm commands.")
    parser.add_argument("dataset", type=Path, help="JSON dataset path; paired arrays are stored beside it")
    parser.add_argument("--name", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--torso", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--serial", default="254322072511")
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is required on the G1") from exc
    if args.bbox[2] <= args.bbox[0] or args.bbox[3] <= args.bbox[1]:
        raise SystemExit("--bbox must have positive area")

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(1000)
        frames = align.process(pipeline.wait_for_frames(1000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D435I did not return paired color and aligned depth")
        rgb = np.asanyarray(color.get_data()).copy()
        z16 = np.asanyarray(depth.get_data()).astype("<u2", copy=True)
        if rgb.shape[:2] != (540, 960) or z16.shape != (540, 960):
            raise RuntimeError(f"unexpected aligned shapes RGB={rgb.shape}, depth={z16.shape}")
        device = profile.get_device()
        rgb_profile = color.profile.as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        scale = float(device.first_depth_sensor().get_depth_scale())
        raw_x1, raw_y1, raw_x2, raw_y2 = (float(item) for item in args.bbox)
        center_x, center_y = (raw_x1 + raw_x2) / 2.0, (raw_y1 + raw_y2) / 2.0
        half_width, half_height = (raw_x2 - raw_x1) * 0.3, (raw_y2 - raw_y1) * 0.3
        x1, x2 = max(0, int(np.floor(center_x - half_width))), min(960, int(np.ceil(center_x + half_width)))
        y1, y2 = max(0, int(np.floor(center_y - half_height))), min(540, int(np.ceil(center_y + half_height)))
        roi = z16[y1:y2, x1:x2].astype(float) * scale
        valid = roi[(roi >= 0.12) & (roi <= 4.0)]
        if len(valid) < 8:
            raise RuntimeError("not enough valid aligned depth pixels in bbox")
        root = args.dataset.parent / (args.dataset.stem + "_frames")
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / f"{args.name}_rgb.npy", rgb)
        np.save(root / f"{args.name}_aligned_depth_z16.npy", z16)
        existing = {"samples": []}
        if args.dataset.exists():
            existing = json.loads(args.dataset.read_text(encoding="utf-8"))
        if any(item.get("name") == args.name for item in existing.get("samples", [])):
            raise RuntimeError(f"duplicate sample name {args.name!r}")
        existing["camera_serial"] = device.get_info(rs.camera_info.serial_number)
        existing["camera_firmware"] = device.get_info(rs.camera_info.firmware_version)
        existing["rgb_profile"] = {"width": 960, "height": 540, "fps": 60, "format": "rgb8"}
        existing["depth_profile"] = {"width": int(depth_profile.width()), "height": int(depth_profile.height()), "fps": 60, "format": "z16", "aligned_to_rgb": True}
        existing["rgb_intrinsics"] = _intrinsics(rgb_profile.get_intrinsics())
        existing["depth_scale"] = scale
        existing.setdefault("samples", []).append({
            "name": args.name, "bbox_xyxy": args.bbox, "torso_truth_m": args.torso,
            "median_aligned_depth_m": float(np.median(valid)), "depth_valid_fraction": float(len(valid) / roi.size),
            "rgb_file": str(root / f"{args.name}_rgb.npy"), "aligned_depth_file": str(root / f"{args.name}_aligned_depth_z16.npy"),
            "captured_unix_s": time.time(),
        })
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        args.dataset.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(existing["samples"][-1], indent=2))
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
