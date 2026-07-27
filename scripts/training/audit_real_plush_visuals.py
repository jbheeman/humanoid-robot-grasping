#!/usr/bin/env python3
"""Audit real moving-plush episodes with structure, contact, and YOLO motion checks.

Raw episodes are never moved or deleted. The output is a set of manifests used
by later conversion, including a contact-bounded training frame range that
excludes human reset activity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from object_tracking.real_visual_qc import (
    classify_episode,
    contact_runs,
    maximum_center_displacement_px,
    structural_reasons,
    training_frame_range,
    visual_reasons,
)


def parse_episode_ids(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def structural_metrics(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((path / "data.json").read_text(encoding="utf-8"))
    frames = payload.get("data", [])
    if not isinstance(frames, list):
        raise ValueError(f"{path}: data must be a list")
    timestamps = [
        int(frame.get("timing", {}).get("monotonic_time_ns", 0))
        for frame in frames
    ]
    deltas = np.diff(np.asarray(timestamps, dtype=np.int64)) / 1e9
    monotonic = bool(len(deltas) and np.all(deltas > 0.0))
    sample_hz = 1.0 / median(deltas) if monotonic else 0.0
    contact = [
        bool(
            frame.get("annotations", {})
            .get("right_palm_contact", {})
            .get("value", False)
        )
        for frame in frames
    ]
    runs = contact_runs(contact)
    invalid_joint_frames = 0
    missing_images = 0
    for frame in frames:
        state = frame.get("states", {}).get("right_arm", {}).get("qpos", [])
        action = frame.get("actions", {}).get("right_arm", {}).get("qpos", [])
        invalid_joint_frames += int(len(state) != 7 or len(action) != 7)
        image = path / str(frame.get("colors", {}).get("color_0", ""))
        missing_images += int(not image.is_file())
    return {
        "frames": len(frames),
        "duration_s": (
            (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
        ),
        "sample_hz": sample_hz,
        "timestamps_monotonic": monotonic,
        "missing_images": missing_images,
        "invalid_joint_frames": invalid_joint_frames,
        "first_contact_frame": runs[0][0] if runs else -1,
        "last_contact_frame": runs[-1][1] - 1 if runs else -1,
        "contact_frames": sum(contact),
        "contact_transitions": int(
            np.count_nonzero(np.diff(np.asarray(contact, dtype=np.int8)))
        ),
    }, frames


def sample_indices(first_contact: int, frame_count: int, sample_hz: float) -> list[int]:
    stop = first_contact if first_contact >= 0 else frame_count - 1
    stride = max(1, round(sample_hz / 5.0))
    indices = set(range(0, stop + 1, stride))
    indices.update(
        index
        for index in range(max(0, stop - 2), min(frame_count, stop + 3))
    )
    indices.add(stop)
    return sorted(indices)


def best_center(result: Any, confidence: float) -> tuple[float, float] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    scores = boxes.conf.detach().cpu().numpy()
    valid = np.flatnonzero(scores >= confidence)
    if not len(valid):
        return None
    selected = int(valid[np.argmax(scores[valid])])
    x1, y1, x2, y2 = boxes.xyxy[selected].detach().cpu().numpy().tolist()
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operator-reject", default="")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    operator_rejects = parse_episode_ids(args.operator_reject)
    model = YOLO(str(args.detector))
    records: list[dict[str, Any]] = []
    for path in sorted(args.dataset_root.glob("episode_*")):
        if not path.is_dir():
            continue
        episode_id = int(path.name.rsplit("_", 1)[-1])
        metrics, frames = structural_metrics(path)
        structural = list(structural_reasons(metrics))
        if not metrics["timestamps_monotonic"]:
            structural.append("timestamps_not_monotonic")
        first_contact = int(metrics["first_contact_frame"])
        visual: tuple[str, ...] = ()
        centers: list[tuple[float, float] | None] = []
        contact_visible = False
        sampled: list[int] = []
        if not structural and episode_id not in operator_rejects:
            sampled = sample_indices(
                first_contact, len(frames), float(metrics["sample_hz"])
            )
            images = [
                str(path / frames[index]["colors"]["color_0"])
                for index in sampled
            ]
            results = model.predict(
                source=images,
                conf=args.confidence,
                device=args.device,
                verbose=False,
                stream=False,
            )
            centers = [best_center(result, args.confidence) for result in results]
            contact_visible = any(
                center is not None
                for index, center in zip(sampled, centers)
                if abs(index - first_contact) <= 2
            )
            visual = visual_reasons(
                centers=centers,
                contact_visible=contact_visible,
            )
        status, reasons = classify_episode(
            operator_rejected=episode_id in operator_rejects,
            structural=structural,
            visual=visual,
        )
        trim = (
            training_frame_range(len(frames), first_contact)
            if first_contact >= 0
            else None
        )
        records.append(
            {
                "episode_id": episode_id,
                "episode": path.name,
                "path": str(path),
                "status": status,
                "reasons": list(reasons),
                "metrics": {
                    **metrics,
                    "sampled_visual_frames": len(sampled),
                    "plush_visible_fraction": (
                        sum(center is not None for center in centers) / len(centers)
                        if centers
                        else None
                    ),
                    "plush_path_displacement_px": (
                        maximum_center_displacement_px(centers)
                        if centers
                        else None
                    ),
                    "plush_visible_at_contact": (
                        contact_visible if centers else None
                    ),
                },
                "training_frame_range": (
                    {"start": trim[0], "end_exclusive": trim[1]} if trim else None
                ),
                "raw_data_modified": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "episode_qc_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_counts = Counter(record["status"] for record in records)
    reason_counts = Counter(
        reason for record in records for reason in record["reasons"]
    )
    summary = {
        "schema_version": 1,
        "dataset_root": str(args.dataset_root),
        "detector": str(args.detector),
        "episodes": len(records),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "operator_reject_ids": sorted(operator_rejects),
        "raw_data_modified": False,
        "human_reset_frames_excluded_by_contact_bounded_trim": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
