#!/usr/bin/env python3
"""Smoke-test v30 horizon normalization and active-motion sampling."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["datasets"]["vla_data"]
    stats = json.loads(args.statistics.read_text(encoding="utf-8"))
    if (
        stats["representation"]["normalization"]["version"]
        != "per_horizon_bounds_q99_v1"
    ):
        raise ValueError("statistics are not v30 per-horizon bounds")
    os.environ["G1_PLUSH_SHARED_STATS"] = str(args.statistics.resolve())
    os.environ.setdefault("G1_PLUSH_REAL_WEIGHT", "8.0")
    import sys

    sys.path.insert(0, str(args.workspace / "unifolm-vla/src"))
    from unifolm_vla.rlds_dataloader.datasets.datasets import RLDSDataset

    window = int(data["window_size"])

    def transform(sample: dict) -> dict:
        return {
            "action": sample["action"][window - 1 :],
            "active_motion": bool(sample["active_motion"]),
            "dataset_name": sample["dataset_name"].decode(),
        }

    dataset = RLDSDataset(
        Path(data["data_root_dir"]),
        data["data_mix"],
        transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=max(args.samples, 256),
        train=True,
        image_aug=bool(data["image_aug"]),
        window_size=window,
        observation_stride=int(data["observation_stride"]),
        active_motion_threshold_m=float(data["active_motion_threshold_m"]),
        inactive_keep_probability=float(data["inactive_keep_probability"]),
        action_representation=data["action_representation"],
        relative_action_statistics=str(args.statistics),
    )
    iterator = iter(dataset)
    counts: Counter[str] = Counter()
    active = 0
    saturation = []
    for _ in range(args.samples):
        sample = next(iterator)
        action = np.asarray(sample["action"])
        if action.shape != (25, 23):
            raise AssertionError(f"unexpected action shape {action.shape}")
        if not np.all(np.isfinite(action)) or np.max(np.abs(action)) > 1.0001:
            raise AssertionError("normalized actions are invalid")
        counts[sample["dataset_name"]] += 1
        active += int(sample["active_motion"])
        saturation.append(float(np.mean(np.abs(action) >= 0.999)))
    print(
        json.dumps(
            {
                "samples": args.samples,
                "sources": counts,
                "active_fraction": active / args.samples,
                "normalized_saturation_fraction": float(np.mean(saturation)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
