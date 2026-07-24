#!/usr/bin/env python3
"""Audit real/synthetic active-window distributions for the v30 contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def episode_motion(right_xyz: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(right_xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("right_xyz must be [T,3]")
    displacement = np.zeros((len(values), horizon), dtype=np.float64)
    valid = np.zeros((len(values), horizon), dtype=bool)
    for step in range(horizon):
        count = len(values) - step - 1
        if count <= 0:
            break
        displacement[:count, step] = np.linalg.norm(
            values[step + 1 :] - values[:count], axis=1
        )
        valid[:count, step] = True
    return displacement, valid


def audit(
    root: Path, manifest: dict, *, horizon: int, threshold: float
) -> dict:
    import h5py

    accumulators: dict[str, list[np.ndarray]] = {"real": [], "sim": []}
    validity: dict[str, list[np.ndarray]] = {"real": [], "sim": []}
    contacts = {"real": 0, "sim": 0}
    episodes = {"real": 0, "sim": 0}
    for entry in manifest["episodes"]:
        if str(entry["split"]).replace("validation", "val") != "train":
            continue
        source = "sim" if entry["source"] == "isaac" else "real"
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path
            if not path.exists():
                path = root.parent.parent / entry["path"]
        with h5py.File(path, "r") as episode:
            ee = np.asarray(episode["observations/ee_qpos"][:], dtype=np.float64)
            displacement, valid = episode_motion(ee[:, 6:9], horizon)
            if "annotations/right_palm_contact" in episode:
                contacts[source] += int(
                    np.count_nonzero(episode["annotations/right_palm_contact"][:])
                )
        accumulators[source].append(displacement)
        validity[source].append(valid)
        episodes[source] += 1

    report = {
        "schema_version": 1,
        "horizon": horizon,
        "active_motion_threshold_m": threshold,
        "sources": {},
    }
    for source in ("real", "sim"):
        values = np.concatenate(accumulators[source])
        valid = np.concatenate(validity[source])
        maximum = np.max(np.where(valid, values, 0.0), axis=1)
        active = maximum >= threshold
        report["sources"][source] = {
            "episodes": episodes[source],
            "windows": int(len(values)),
            "active_windows": int(np.count_nonzero(active)),
            "active_fraction": float(np.mean(active)),
            "contact_frames": contacts[source],
            "per_horizon_mean_m": [
                float(np.mean(values[:, index][valid[:, index]]))
                for index in range(horizon)
            ],
            "per_horizon_q99_m": [
                float(np.quantile(values[:, index][valid[:, index]], 0.99))
                for index in range(horizon)
            ],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--active-threshold-m", type=float, default=0.01)
    args = parser.parse_args()
    manifest = json.loads(
        (args.canonical_root / "CANONICAL_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    payload = audit(
        args.canonical_root,
        manifest,
        horizon=args.horizon,
        threshold=args.active_threshold_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload["sources"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
