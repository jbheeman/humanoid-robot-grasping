#!/usr/bin/env python3
"""Build the G1 bunny-stop TFDS/RLDS dataset without TFDS CLI extras."""

from __future__ import annotations

import argparse
import importlib.util
import os
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
    spec = importlib.util.spec_from_file_location("g1_bunny_stop", args.builder)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TFDS builder from {args.builder}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    builder = module.G1BunnyStop(data_dir=args.output)
    builder.download_and_prepare()
    print(f"built {builder.info.full_name} at {builder.data_path}")


if __name__ == "__main__":
    main()
