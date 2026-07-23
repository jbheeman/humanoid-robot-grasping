#!/usr/bin/env python3
"""Build train-only absolute Pose23 stats for achieved future-state targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_relative_pose23_statistics import (
    atomic_json,
    resolve_episode,
    sha256,
    summarize,
)
from object_tracking.unifolm_relative_actions import pose17_to_pose23
from object_tracking.vla_target_alignment import (
    FUTURE_STATE_TARGET_V1,
    align_achieved_future_state,
    alignment_metadata,
)


def main() -> int:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required to read canonical episodes") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--base-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-state-lookahead", type=int, required=True)
    parser.add_argument("--real-weight", type=float, default=0.75)
    args = parser.parse_args()
    if args.future_state_lookahead < 1:
        raise ValueError("--future-state-lookahead must be positive")
    if not 0.0 < args.real_weight < 1.0:
        raise ValueError("--real-weight must be between zero and one")

    manifest_path = args.canonical_root / "CANONICAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_stats = json.loads(args.base_stats.read_text(encoding="utf-8"))
    by_source: dict[str, list[np.ndarray]] = {"sim": [], "real": []}
    counts = {
        "sim": {"trajectories": 0, "targets": 0},
        "real": {"trajectories": 0, "targets": 0},
    }
    for entry in manifest["episodes"]:
        if str(entry["split"]).replace("validation", "val") != "train":
            continue
        source = "sim" if entry["source"] == "isaac" else "real"
        path = resolve_episode(args.canonical_root, entry["path"])
        with h5py.File(path, "r") as episode:
            states = pose17_to_pose23(
                np.asarray(episode["observations/ee_qpos"][:])
            )
        targets = align_achieved_future_state(
            states, args.future_state_lookahead
        ).targets
        by_source[source].append(targets)
        counts[source]["trajectories"] += 1
        counts[source]["targets"] += len(targets)

    values_parts = []
    weight_parts = []
    for source, mass in (
        ("sim", 1.0 - args.real_weight),
        ("real", args.real_weight),
    ):
        values = np.concatenate(by_source[source]).astype(np.float64)
        values_parts.append(values)
        weight_parts.append(
            np.full(len(values), mass / len(values), dtype=np.float64)
        )
    values = np.concatenate(values_parts)
    weights = np.concatenate(weight_parts)
    payload = {
        "action": summarize(values, weights),
        "proprio": base_stats["proprio"],
        "num_trajectories": sum(v["trajectories"] for v in counts.values()),
        "num_transitions": len(values),
        "representation": {
            "version": "absolute_pose23",
            "target_alignment": alignment_metadata(
                args.future_state_lookahead
            ),
        },
        "provenance": {
            "train_only": True,
            "canonical_manifest_sha256": sha256(manifest_path),
            "base_statistics_sha256": sha256(args.base_stats),
            "source_weights": {
                "sim": 1.0 - args.real_weight,
                "real": args.real_weight,
            },
            "source_counts": counts,
            "future_state_target_version": FUTURE_STATE_TARGET_V1,
            "weighting": "equal source mass then equal target mass within source",
        },
    }
    atomic_json(args.output, payload)
    print(
        "FUTURE_STATE_ABSOLUTE_STATS_READY",
        f"output={args.output}",
        f"targets={len(values)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
