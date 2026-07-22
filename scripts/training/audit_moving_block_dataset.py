#!/usr/bin/env python3
"""Fail closed on moving-rabbit dataset physics, storage, and diversity gates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")


def average_hash(image: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(image).convert("L").resize((16, 16), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    return values >= values.mean()


def value(root: h5py.File, name: str, default: float) -> float:
    return float(root.attrs.get(name, default))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mixed-language", action="store_true")
    args = parser.parse_args()
    paths = sorted(
        path
        for pattern in ("case_*/episode_*.hdf5", "worker*/episode_*.hdf5")
        for path in args.dataset.glob(pattern)
        if path.is_file()
    )
    errors = []
    if len(paths) != args.expected_count:
        errors.append(f"accepted_count={len(paths)} expected={args.expected_count}")
    records = []
    hashes = []
    exact_hashes = []
    languages = set()
    for path in paths:
        with h5py.File(path, "r") as root:
            language = root["language_raw"][()]
            language = language.decode() if isinstance(language, bytes) else str(language)
            languages.add(language)
            aliases = [h5py.h5o.get_info(root[f"observations/images/{name}"].id).addr for name in CAMERAS]
            first_contact = int(root.attrs.get("first_contact_frame", -1))
            image_index = first_contact if first_contact >= 0 else len(root["timestamp"]) // 2
            contact_image = np.asarray(
                root["observations/images/cam_left_high"][image_index]
            )
            hashes.append(average_hash(contact_image))
            exact_hashes.append(hashlib.sha256(contact_image.tobytes()).hexdigest())
            checks = {
                "contact": value(root, "contact_frames", 0) >= 1,
                "clearance": value(root, "min_enabled_palm_clearance_m", -1) >= 0.05,
                "left_force": value(root, "maximum_left_hand_object_force_n", float("inf")) <= 0.25,
                "right_table_force": value(root, "maximum_right_hand_table_force_n", float("inf")) <= 0.50,
                "contact_extension": value(root, "actual_palm_extension_at_contact_m", -1)
                >= value(root, "minimum_contact_extension_m", 0.12),
                "absolute_stop": value(root, "post_contact_path_speed_m_s", float("inf")) <= 0.035,
                "relative_stop": value(root, "stop_fraction", -float("inf")) >= 0.70,
                "postcontact_spin": value(root, "maximum_postcontact_spin_rad_s", float("inf"))
                <= value(root, "maximum_allowed_postcontact_spin_rad_s", 0.35),
                "on_table": not bool(root.attrs.get("table_exit", True)),
                "single_physical_camera": len(set(aliases)) == 1,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append(f"{path.name}:{','.join(failed)}")
            records.append(
                {
                    "path": str(path),
                    "seed": int(root.attrs["trajectory_seed"]),
                    "camera": str(root.attrs["camera_variant"]),
                    "speed_m_s": value(root, "launch_speed_m_s", 0),
                    "heading_deg": value(root, "launch_heading_deg", 0),
                    "object_initial_yaw_deg": value(root, "object_initial_yaw_deg", 0),
                    "contact_frames": int(root.attrs["contact_frames"]),
                    "post_contact_speed_m_s": value(root, "post_contact_path_speed_m_s", 0),
                    "maximum_postcontact_spin_rad_s": value(
                        root, "maximum_postcontact_spin_rad_s", float("inf")
                    ),
                    "minimum_clearance_m": value(root, "min_enabled_palm_clearance_m", 0),
                    "checks": checks,
                }
            )
    if len(languages) > 1 and not args.allow_mixed_language:
        errors.append(f"mixed_language={sorted(languages)}")
    distances = [
        int(np.count_nonzero(left != right))
        for left, right in itertools.combinations(hashes, 2)
    ]
    perceptual_hash_collision_pairs = sum(distance == 0 for distance in distances)
    exact_duplicate_pairs = sum(
        left == right for left, right in itertools.combinations(exact_hashes, 2)
    )
    if exact_duplicate_pairs:
        errors.append(f"exact_duplicate_contact_views={exact_duplicate_pairs}")
    heading_span_deg = None
    yaw_bin_count = None
    if records:
        headings = [record["heading_deg"] for record in records]
        heading_span_deg = max(headings) - min(headings)
        yaw_bins = {
            int(record["object_initial_yaw_deg"] % 360.0 // 30.0)
            for record in records
        }
        yaw_bin_count = len(yaw_bins)
        # Small one-off smoke runs should remain usable, while review and full
        # datasets must demonstrate genuine path and appearance diversity.
        if len(records) >= 12:
            if heading_span_deg < 25.0:
                errors.append(f"heading_span_deg={heading_span_deg:.3f} expected>=25")
            if yaw_bin_count < 8:
                errors.append(f"object_yaw_bins={yaw_bin_count} expected>=8")
    report = {
        "passed": not errors,
        "errors": errors,
        "accepted_count": len(paths),
        "languages": sorted(languages),
        "exact_duplicate_contact_views": exact_duplicate_pairs,
        "perceptual_hash_collision_pairs": perceptual_hash_collision_pairs,
        "minimum_contact_view_hash_distance": min(distances) if distances else None,
        "heading_span_deg": heading_span_deg,
        "object_yaw_30deg_bin_count": yaw_bin_count,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "errors", "accepted_count")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
