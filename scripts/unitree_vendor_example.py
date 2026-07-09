#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = {
    "g1_loco": "scripts/unitree_examples/g1/high_level/g1_loco_client_example.py",
    "g1_arm_action": "scripts/unitree_examples/g1/high_level/g1_arm_action_example.py",
    "g1_arm5": "scripts/unitree_examples/g1/high_level/g1_arm5_sdk_dds_example.py",
    "g1_arm7": "scripts/unitree_examples/g1/high_level/g1_arm7_sdk_dds_example.py",
    "g1_low_level": "scripts/unitree_examples/g1/low_level/g1_low_level_example.py",
    "motion_switcher": "scripts/unitree_examples/motionSwitcher/motion_switcher_example.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run copied Unitree SDK2 example files verbatim.")
    parser.add_argument("example", nargs="?", choices=("list", *EXAMPLES), default="list")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument(
        "--no-interface-arg",
        action="store_true",
        help="Do not append the interface argument. Most Unitree examples expect it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.example == "list":
        for name, path in EXAMPLES.items():
            print(f"{name}: {path}")
        return 0

    script = REPO_ROOT / EXAMPLES[args.example]
    command = [sys.executable, str(script)]
    if not args.no_interface_arg:
        command.append(args.interface)

    print("Running copied Unitree example verbatim:")
    print(" ".join(command))
    return subprocess.call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
