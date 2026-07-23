#!/usr/bin/env python3
"""Programmatically materialize plush-touch TFDS without Apache Beam."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorflow_datasets as tfds

from g1_plush_touch_real.g1_plush_touch_real import G1PlushTouchReal
from g1_plush_touch_sim.g1_plush_touch_sim import G1PlushTouchSim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sim", "real"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument(
        "--future-state-lookahead",
        type=int,
        default=0,
        help=(
            "replace recorded commands with achieved state[t+N] and drop the "
            "final N unlabeled observations"
        ),
    )
    args = parser.parse_args()
    if args.future_state_lookahead < 0:
        raise ValueError("--future-state-lookahead cannot be negative")
    os.environ["G1_PLUSH_FUTURE_STATE_LOOKAHEAD"] = str(
        args.future_state_lookahead
    )
    builder_class = G1PlushTouchSim if args.source == "sim" else G1PlushTouchReal
    builder = builder_class(data_dir=str(args.data_dir))
    builder.download_and_prepare(
        download_dir=str(args.download_dir),
        download_config=tfds.download.DownloadConfig(try_download_gcs=False),
    )
    print(
        "TFDS_BUILD_PASS",
        f"name={builder.name}",
        f"version={builder.version}",
        f"splits={dict(builder.info.splits)}",
        f"future_state_lookahead={args.future_state_lookahead}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
