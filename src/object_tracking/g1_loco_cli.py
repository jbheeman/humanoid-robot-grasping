from __future__ import annotations

import argparse
from pathlib import Path

from object_tracking.unitree_g1 import G1LocoSdk2Client, UnitreeG1Error, parse_velocity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send Unitree SDK2 G1 loco commands.")
    parser.add_argument("robot_ip", nargs="?", help="Robot IP address.")
    parser.add_argument(
        "--robot-ip",
        dest="robot_ip_option",
        help="Robot IP address. Same as positional robot_ip.",
    )
    parser.add_argument(
        "--network-interface",
        help="Override local interface. Usually omit this and pass robot_ip instead.",
    )
    parser.add_argument(
        "--sdk2-path",
        type=Path,
        default=Path("~/Documents/unitree_sdk2").expanduser(),
        help="Path to the Unitree SDK2 checkout.",
    )
    parser.add_argument(
        "--loco-binary",
        type=Path,
        help="Path to a built g1_loco_client binary.",
    )
    parser.add_argument(
        "command",
        choices=("get_fsm_id", "start", "stand_up", "balance_stand", "stop_move", "damp", "move"),
        help="G1 loco command to send.",
    )
    parser.add_argument(
        "--velocity",
        type=parse_velocity,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Required for command=move.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    robot_ip = args.robot_ip_option or args.robot_ip
    if robot_ip is None and args.network_interface is None:
        raise SystemExit("Pass robot_ip, --robot-ip, or --network-interface.")

    try:
        client = G1LocoSdk2Client(
            network_interface=args.network_interface,
            robot_ip=robot_ip,
            sdk2_path=args.sdk2_path,
            loco_binary=args.loco_binary,
        )
        if args.command == "move":
            if args.velocity is None:
                raise SystemExit('move requires --velocity "VX VY OMEGA [DURATION]"')
            result = client.move(*args.velocity)
        else:
            result = client.command(args.command)
    except UnitreeG1Error as exc:
        raise SystemExit(str(exc)) from exc

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
