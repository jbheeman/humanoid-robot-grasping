#!/usr/bin/env python3
"""Rotate one Dex3-1 finger joint by a set number of degrees, over DDS.

This is a hardware test helper. It commands ONLY a single motor on ONE Dex3-1
hand (``rt/dex3/{left,right}/cmd``) and touches nothing else on the robot, so it
is the smallest possible "move a joint" smoke test.

There is no per-joint control for the arms/legs in this project (loco only
exposes the high-level SportClient verbs), so "rotate a joint" here means a
finger joint on the hand.

Run it via uv so the loco extra (``unitree_sdk2py``) is importable and the
``object_tracking`` package is on the path:

    uv run python scripts/rotate_joint.py 192.168.0.4 --side right --motor 0 --degrees 15

Inspect the live angles first (no motion):

    uv run python scripts/rotate_joint.py 192.168.0.4 --side right --check
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from object_tracking.g1_grasp import DEX3_NUM_MOTORS, G1HandController
from object_tracking.unitree_g1 import UnitreeG1Error, resolve_network_interface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate a single Dex3-1 finger joint by N degrees (relative to its "
            "current angle) over rt/dex3/<side>/cmd. Nothing else on the robot "
            "is commanded. Stand the robot up first."
        )
    )
    parser.add_argument("robot_ip", help="Robot IP address (used to find the local interface).")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        required=True,
        help="Which hand the joint is on.",
    )
    parser.add_argument(
        "--motor",
        type=int,
        default=0,
        help=f"Finger motor index, 0..{DEX3_NUM_MOTORS - 1} (default: 0).",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=15.0,
        help="Degrees to rotate the joint, relative to its current angle (default: 15).",
    )
    parser.add_argument("--kp", type=float, default=0.6, help="Position gain (keep gentle).")
    parser.add_argument("--kd", type=float, default=0.05, help="Damping gain.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=1.0,
        help="How long to hold the new target, at 50 Hz (default: 1.0).",
    )
    parser.add_argument(
        "--network-interface",
        help="Override local interface. Omit to resolve it from robot_ip via `ip route` (Linux).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Seconds to wait before moving, so you can reach the e-stop (default: 5).",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the safety countdown.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the live per-motor angles + torque and exit without moving.",
    )
    return parser


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print("About to move a G1 finger joint. Keep clear and be ready to e-stop.", file=sys.stderr)
    for remaining in range(seconds, 0, -1):
        print(f"  starting in {remaining}...", file=sys.stderr, flush=True)
        time.sleep(1.0)


def main() -> None:
    args = build_parser().parse_args()

    if not 0 <= args.motor < DEX3_NUM_MOTORS:
        print(f"--motor must be in 0..{DEX3_NUM_MOTORS - 1}", file=sys.stderr)
        raise SystemExit(2)

    try:
        interface = args.network_interface or resolve_network_interface(args.robot_ip)
        hand = G1HandController(interface, side=args.side)
        hand.wait_for_state()

        angles = [q for (q, _tau) in hand.read_side(args.side)]

        if args.check:
            print(f"Connected on interface {interface}. Live {args.side} Dex3-1 state:")
            for i, (q, tau) in enumerate(hand.read_side(args.side)):
                print(f"  motor {i}: q={q:+.4f} rad ({math.degrees(q):+.2f} deg)  tau_est={tau:+.4f}")
            return

        target = list(angles)
        target[args.motor] += math.radians(args.degrees)
        print(
            f"{args.side} motor {args.motor}: "
            f"{math.degrees(angles[args.motor]):+.2f} deg -> {math.degrees(target[args.motor]):+.2f} deg"
        )

        _countdown(0 if args.yes else args.countdown)

        steps = max(1, int(args.seconds / 0.02))
        for _ in range(steps):
            hand._send(args.side, tuple(target), args.kp, args.kd)
            time.sleep(0.02)
        print("Done.")
    except UnitreeG1Error as exc:
        print(f"Rotate failed: {exc}", file=sys.stderr)
        print("Tip: run `uv run loco <ip> --diagnose` to check network + SDK setup.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
