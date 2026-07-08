from __future__ import annotations

import argparse
import json
import subprocess 
import sys

from object_tracking.unitree_g1 import G1LocoSdk2Client, UnitreeG1Error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send Unitree G1 commands via unitree-sdk2 Python package (version 1.0.1)."
    )
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
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for each SDK command.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("stand_up", "balance_stand", "stop_move", "damp", "move", "move_arms_up"),
        help="G1 loco command to send. Omit and pass --diagnose for diagnostics only.",
    )
    parser.add_argument(
        "--velocity",
        type=str,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Required for command=move.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print SDK route/proxy diagnostics and exit.",
    )
    return parser


def _parse_velocity(raw: str) -> tuple[float, float, float, float | None]:
    parts = raw.split()
    if len(parts) not in (3, 4):
        raise ValueError("Expected --velocity as 'vx vy omega' or 'vx vy omega duration'.")
    vx, vy, omega = (float(part) for part in parts[:3])
    duration = float(parts[3]) if len(parts) == 4 else None
    return vx, vy, omega, duration


def _diagnose_route(robot_ip: str | None) -> dict[str, object]:
    if robot_ip is None:
        return {"ok": False, "error": "robot_ip not provided"}
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip(),
            "raw": completed.stdout.strip(),
        }

    tokens = completed.stdout.split()
    interface: str | None = None
    if "dev" in tokens:
        idx = tokens.index("dev") + 1
        if idx < len(tokens):
            interface = tokens[idx]

    return {
        "ok": True,
        "raw": completed.stdout.strip(),
        "interface": interface,
    }


def _sdk2_python_presence() -> dict[str, object]:
    try:
        from importlib.metadata import version

        return {"available": True, "module": "unitree-sdk2", "version": version("unitree-sdk2")}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _diagnose_client(
    robot_ip: str | None,
    network_interface: str | None,
    timeout_s: float,
) -> None:
    if robot_ip is None and network_interface is None:
        print("Pass robot_ip/--robot-ip or --network-interface to run diagnostics.", file=sys.stderr)
        raise SystemExit(1)

    route = _diagnose_route(robot_ip)
    resolved_interface = network_interface
    if route.get("ok") and route.get("interface") and network_interface is None:
        resolved_interface = route["interface"]

    report = {
        "robot_ip": robot_ip,
        "timeout_s": timeout_s,
        "route": route,
        "resolved_interface": resolved_interface,
        "unitree_sdk2": _sdk2_python_presence(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    robot_ip = args.robot_ip_option or args.robot_ip

    if args.diagnose:
        _diagnose_client(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            timeout_s=args.timeout,
        )
        return

    if args.command is None:
        print(
            "Missing command. Use one of: stand_up, balance_stand, stop_move, damp, move, move_arms_up.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if robot_ip is None and args.network_interface is None:
        print("Pass robot_ip/--robot-ip, or --network-interface.", file=sys.stderr)
        raise SystemExit(1)

    if args.command == "move":
        if args.velocity is None:
            print('move requires --velocity "vx vy omega [duration]"', file=sys.stderr)
            raise SystemExit(1)
        try:
            vx, vy, omega, duration = _parse_velocity(args.velocity)
        except ValueError as exc:
            print(f"Invalid --velocity: {exc}", file=sys.stderr)
            raise SystemExit(1)

    try:
        client = G1LocoSdk2Client(
            network_interface=args.network_interface,
            robot_ip=robot_ip,
            timeout_s=args.timeout,
        )
        if args.command == "move":
            result = client.move(vx, vy, omega, duration)
        elif args.command == "move_arms_up":
            print("Moving arms to the forward test pose. Press Ctrl-C to stop holding.", file=sys.stderr)
            result = client.move_arms_up()
        else:
            result = client.command(args.command)
    except UnitreeG1Error as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        print("Tip: run `uv run loco <ip> --diagnose` to check network + python sdk setup.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
