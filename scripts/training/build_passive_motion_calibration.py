#!/usr/bin/env python3
"""Measure motion/blur statistics from robot-stationary bunny-slide clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def analyze(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"video has no valid FPS metadata: {path}")
    previous = None
    flows: list[float] = []
    blur: list[float] = []
    luminance: list[float] = []
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        luminance.append(float(gray.mean()))
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            magnitude = np.linalg.norm(flow, axis=2)
            flows.append(float(np.percentile(magnitude, 90)))
        previous = gray
        count += 1
    capture.release()
    if count < max(10, round(fps * 0.5)):
        raise ValueError(f"clip is too short for calibration: {path}")
    return {
        "path": str(path),
        "fps": fps,
        "frames": count,
        "duration_s": count / fps,
        "flow_p90_px_per_frame_median": float(np.median(flows)),
        "flow_p90_px_per_frame_p95": float(np.percentile(flows, 95)),
        "laplacian_variance_median": float(np.median(blur)),
        "laplacian_variance_p05": float(np.percentile(blur, 5)),
        "luminance_mean": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-moving-flow-px", type=float, default=0.35)
    args = parser.parse_args()
    records = [analyze(path) for path in args.clips]
    if len(records) < 3:
        raise SystemExit("provide at least three passive moving-bunny clips")
    median_motion = float(np.median([item["flow_p90_px_per_frame_median"] for item in records]))
    if median_motion < args.minimum_moving_flow_px:
        raise SystemExit(
            f"clips do not contain enough motion ({median_motion:.3f} px/frame); "
            "slide the bunny while the robot remains stationary"
        )
    report = {
        "capture_contract": {
            "robot_stationary": True,
            "source_camera_fps": 60.0,
            "policy_sample_fps": 30.0,
            "use_training_split_only": True,
        },
        "clip_count": len(records),
        "median_motion_px_per_frame": median_motion,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"clip_count": len(records), "median_motion": median_motion}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
