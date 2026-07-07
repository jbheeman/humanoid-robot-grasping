from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from object_tracking.g1_grasp import (
    DEFAULT_POSES,
    GraspConfig,
    G1ArmController,
    joints_for_side,
    run_grasp,
)
from object_tracking.logging import JsonlLogger
from object_tracking.unitree_g1 import UnitreeG1Error, resolve_network_interface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make the Unitree G1 grasp an object steadily using only the arms "
            "and hands (rt/arm_sdk + Dex3-1). Stand the robot up first: "
            "`uv run loco <ip> stand_up` then `uv run loco <ip> balance_stand`."
        )
    )
    parser.add_argument("robot_ip", help="Robot IP address.")
    parser.add_argument(
        "--network-interface",
        help="Override local interface. Usually omit and let it resolve from robot_ip.",
    )
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
        help="Which arm(s)/hand(s) to use (default: both).",
    )
    parser.add_argument(
        "--hand",
        choices=("dex3", "none"),
        default="dex3",
        help="Hand hardware. 'none' commands the arms only (default: dex3).",
    )
    parser.add_argument(
        "--move-time",
        type=float,
        default=3.0,
        help="Seconds for each arm move (reach/ready/lift). Slower is safer.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="Seconds to hold the grasp steadily. Ignored if --hold-forever is set.",
    )
    parser.add_argument(
        "--hold-forever",
        action="store_true",
        help="Hold the grasp until Ctrl+C, then release gracefully.",
    )
    parser.add_argument(
        "--lift",
        action="store_true",
        help="Raise the object slightly after closing the hands.",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Do not move the arms back home before releasing on exit.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=60.0,
        help="Arm position gain. Higher = stiffer/firmer hold (default: 60).",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=1.5,
        help="Arm damping gain (default: 1.5).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Seconds to wait before moving, so you can reach the e-stop (default: 5).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety countdown.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Connect, print the live arm joint angles, and exit without moving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned sequence without publishing any robot commands.",
    )
    parser.add_argument(
        "--output",
        default="runs/grasp",
        help="Directory for the JSONL event log (default: runs/grasp).",
    )
    return parser


def _run_check(robot_ip: str, network_interface: str | None, side: str) -> int:
    interface = network_interface or resolve_network_interface(robot_ip)
    controller = G1ArmController(interface, joints=joints_for_side(side))
    controller.wait_for_state()
    angles = controller.current_arm_q()
    print(f"Connected on interface {interface}. Live arm joint angles (rad):")
    for joint in joints_for_side(side):
        print(f"  joint {joint:2d}: {angles[joint]:+.4f}")
    return 0


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(
        "About to move the G1 arms/hands. Keep clear and be ready to e-stop.",
        file=sys.stderr,
    )
    for remaining in range(seconds, 0, -1):
        print(f"  starting in {remaining}...", file=sys.stderr, flush=True)
        time.sleep(1.0)


def main() -> None:
    args = build_parser().parse_args()

    try:
        if args.check:
            raise SystemExit(_run_check(args.robot_ip, args.network_interface, args.side))

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "grasp.jsonl"

        config = GraspConfig(
            robot_ip=args.robot_ip,
            network_interface=args.network_interface,
            side=args.side,
            use_hands=args.hand != "none",
            poses=DEFAULT_POSES,
            move_time_s=args.move_time,
            hold_seconds=None if args.hold_forever else args.hold_seconds,
            do_lift=args.lift,
            return_home=not args.no_return,
            kp=args.kp,
            kd=args.kd,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print("[dry-run] No commands will be sent to the robot.")
        else:
            _countdown(0 if args.yes else args.countdown)

        with JsonlLogger(log_path) as logger:

            def on_event(name: str, detail: dict[str, object]) -> None:
                record = {"event": name, "timestamp": time.time()}
                record.update(detail)
                logger.write(record)
                print(f"[grasp] {name} {detail if detail else ''}".rstrip())

            run_grasp(config, on_event=on_event)

        print(f"Wrote grasp event log to {log_path}")
    except UnitreeG1Error as exc:
        print(f"Grasp failed: {exc}", file=sys.stderr)
        print("Tip: run `uv run loco <ip> --diagnose` to check network + SDK setup.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
