#!/usr/bin/env python3
"""Build the G1 bunny-stop TFDS/RLDS dataset without TFDS CLI extras."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Glob for episode HDF5 files")
    parser.add_argument("--output", required=True, help="TFDS output directory")
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rlds" / "g1_bunny_stop.py",
    )
    args = parser.parse_args()

    os.environ["G1_BUNNY_HDF5_GLOB"] = args.input
    builder_path = args.builder.resolve()
    sys.path.insert(0, str(builder_path.parent))
    module = importlib.import_module(builder_path.stem)

    builder = module.G1BunnyStop(data_dir=args.output)
    builder.download_and_prepare()
    print(f"built {builder.info.full_name} at {builder.data_path}")


if __name__ == "__main__":
    main()
