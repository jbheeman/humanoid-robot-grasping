"""Analyze a local D435I localization dataset without any robot actuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arm_tracking.geometry import CameraIntrinsics
from .arm_tracking.localization import LocalizationObservation, solve_localization


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list):
        raise ValueError("dataset must be a JSON object with a samples list")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate D435I RGB/depth localization; never commands the robot.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--fit", nargs="+", required=True, metavar="NAME")
    parser.add_argument("--validation", nargs="+", required=True, metavar="NAME")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = _load(args.dataset)
    intrinsics = CameraIntrinsics.from_dict(dataset["rgb_intrinsics"])
    samples = []
    for value in dataset["samples"]:
        bbox = value["bbox_xyxy"]
        samples.append(
            LocalizationObservation(
                name=str(value["name"]),
                pixel_xy=((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0),
                depth_m=float(value["median_aligned_depth_m"]),
                torso_truth_m=tuple(float(item) for item in value["torso_truth_m"]),
            )
        )
    report = solve_localization(samples, intrinsics, fit_names=args.fit, validation_names=args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
