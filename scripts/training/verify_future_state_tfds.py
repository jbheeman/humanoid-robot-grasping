#!/usr/bin/env python3
"""Verify materialized TFDS future-state targets against canonical HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds

from object_tracking.vla_target_alignment import FUTURE_STATE_TARGET_V1


def verify_episode(
    episode: dict,
    *,
    expected_lookahead: int,
) -> tuple[str, int]:
    metadata = episode["episode_metadata"]
    source_path = Path(metadata["file_path"].numpy().decode("utf-8"))
    version = metadata["action_target_version"].numpy().decode("utf-8")
    lookahead = int(metadata["target_lookahead_frames"].numpy())
    if version != FUTURE_STATE_TARGET_V1 or lookahead != expected_lookahead:
        raise AssertionError(
            f"{source_path}: metadata {version}/{lookahead}, expected "
            f"{FUTURE_STATE_TARGET_V1}/{expected_lookahead}"
        )
    steps = episode["steps"]
    state = np.asarray(steps["observation"]["state"])
    ee_state = np.asarray(steps["observation"]["ee_state"])
    action = np.asarray(steps["action"])
    ee_action = np.asarray(steps["ee_action"])
    with h5py.File(source_path, "r") as source:
        source_state = np.asarray(source["observations/qpos"][:])
        source_ee_state = np.asarray(source["observations/ee_qpos"][:])
    expected_count = len(source_state) - expected_lookahead
    if len(state) != expected_count:
        raise AssertionError(
            f"{source_path}: {len(state)} transitions, expected {expected_count}"
        )
    np.testing.assert_allclose(state, source_state[:-expected_lookahead])
    np.testing.assert_allclose(action, source_state[expected_lookahead:])
    np.testing.assert_allclose(ee_state, source_ee_state[:-expected_lookahead])
    np.testing.assert_allclose(
        ee_action, source_ee_state[expected_lookahead:]
    )
    return str(source_path), expected_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, required=True)
    parser.add_argument("--episodes-per-source", type=int, default=3)
    args = parser.parse_args()
    if args.lookahead < 1 or args.episodes_per_source < 1:
        raise ValueError("lookahead and episodes-per-source must be positive")

    checked = []
    for builder_name in ("g1_plush_touch_real", "g1_plush_touch_sim"):
        builder_dir = args.data_root / builder_name / "1.0.0"
        builder = tfds.builder_from_directory(str(builder_dir))
        dataset = builder.as_dataset(split="train")
        for episode in dataset.take(args.episodes_per_source):
            path, transitions = verify_episode(
                episode, expected_lookahead=args.lookahead
            )
            checked.append(
                {
                    "builder": builder_name,
                    "path": path,
                    "transitions": transitions,
                }
            )
    print(
        "FUTURE_STATE_TFDS_VERIFY_PASS",
        f"lookahead={args.lookahead}",
        f"episodes={len(checked)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
