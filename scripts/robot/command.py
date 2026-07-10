#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib import request
from urllib.error import HTTPError, URLError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_HOST = "192.168.0.213"
DEFAULT_PORT = 8765
DEFAULT_INTERFACE = "wlan0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unitree G1 command helper for bridge and bounded tests.")
    parser.add_argument("--host", default=DEFAULT_ROBOT_HOST, help=f"Robot Wi-Fi IP. Default: {DEFAULT_ROBOT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the robot-local HTTP motion bridge.")
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--interface", default=DEFAULT_INTERFACE)
    serve.add_argument("--domain-id", type=int, default=0)
    serve.add_argument("--loco-service-name", default="sport", choices=("auto", "sport", "ai_sport"))
    serve.add_argument("--dds-config-mode", default="no_trace")
    serve.add_argument("--read-only", action="store_true")

    subparsers.add_parser("health", help="Check whether the robot-local HTTP bridge is reachable.")
    subparsers.add_parser("probe", help="Run read-only loco RPC probes through the bridge.")
    subparsers.add_parser("mode", help="Read the Unitree motion-switcher mode through the bridge.")
    subparsers.add_parser("stop", help="Send StopMove through the bridge.")
    subparsers.add_parser("estop", help="Alias for stop. This is software stop, not hardware e-stop.")
    subparsers.add_parser("smoke-move", help="Run the smallest guarded forward movement test.")
    arms = subparsers.add_parser("arms-up", help="Run a bounded arm raise through the bridge.")
    arms.add_argument("amount", nargs="?", type=float, default=0.15)
    arms.add_argument("--ramp", type=float, default=1.5)
    arms.add_argument("--hold", type=float, default=1.0)

    forward = subparsers.add_parser("forward", help="Run one small forward movement through the bridge.")
    forward.add_argument("speed", nargs="?", type=float, default=0.05)
    forward.add_argument("--duration", type=float, default=0.4)
    forward.add_argument("--ramp", type=float, default=0.15)

    move = subparsers.add_parser("move", help="Run one bounded velocity command through the bridge.")
    move.add_argument("--vx", type=float, default=0.0)
    move.add_argument("--vy", type=float, default=0.0)
    move.add_argument("--omega", "--wz", dest="omega", type=float, default=0.0)
    move.add_argument("--duration", type=float, default=0.4)
    move.add_argument("--ramp", type=float, default=0.15)

    shoulder = subparsers.add_parser(
        "shoulder-pitch",
        help="Robot-local low-level shoulder pitch test. Run this on the robot.",
    )
    shoulder.add_argument("--interface", default=DEFAULT_INTERFACE)
    shoulder.add_argument("--domain-id", type=int, default=0)
    shoulder.add_argument("--side", choices=("left", "right", "both"), default="both")
    shoulder.add_argument("--delta", type=float, default=0.25)
    shoulder.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    shoulder.add_argument("--direction", choices=("positive", "negative"), default=None)
    shoulder.add_argument("--left-sign", type=int, choices=(-1, 1), default=None)
    shoulder.add_argument("--right-sign", type=int, choices=(-1, 1), default=None)
    shoulder.add_argument("--ramp-seconds", type=float, default=3.0)
    shoulder.add_argument("--hold-seconds", type=float, default=5.0)
    shoulder.add_argument("--kp-arm", type=float, default=25.0)
    shoulder.add_argument("--kd-arm", type=float, default=1.0)
    shoulder.add_argument("--kp-body", type=float, default=40.0)
    shoulder.add_argument("--kd-body", type=float, default=1.0)
    shoulder.add_argument("--smoke", action="store_true")

    return parser


def bridge_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def send_json(url: str, payload: dict[str, object] | None = None) -> tuple[int, str]:
    if payload is None:
        req = request.Request(url, method="GET")
    else:
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with request.urlopen(req, timeout=30.0) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except URLError as exc:
        return 0, json.dumps({"ok": False, "error": str(exc)}, indent=2)


def run_script(script: str, args: list[str]) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "robot" / script), *args]
    return subprocess.call(command)


def serve(args: argparse.Namespace) -> int:
    command = [
        "--host",
        args.bind,
        "--port",
        str(args.port),
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--loco-service-name",
        args.loco_service_name,
        "--dds-config-mode",
        args.dds_config_mode,
    ]
    if args.read_only:
        command.append("--read-only")
    return run_script("motion_bridge.py", command)


def shoulder_pitch(args: argparse.Namespace) -> int:
    command = [
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--side",
        args.side,
        "--delta",
        str(args.delta),
        "--sign",
        str(args.sign),
        "--ramp-seconds",
        str(args.ramp_seconds),
        "--hold-seconds",
        str(args.hold_seconds),
        "--kp-arm",
        str(args.kp_arm),
        "--kd-arm",
        str(args.kd_arm),
        "--kp-body",
        str(args.kp_body),
        "--kd-body",
        str(args.kd_body),
    ]
    if args.direction:
        command.extend(["--direction", args.direction])
    if args.left_sign is not None:
        command.extend(["--left-sign", str(args.left_sign)])
    if args.right_sign is not None:
        command.extend(["--right-sign", str(args.right_sign)])
    if args.smoke:
        command.append("--smoke")
    return run_script("arms_forward.py", command)


def print_response(status: int, body: str) -> int:
    print(body, end="" if body.endswith("\n") else "\n")
    return 0 if 200 <= status < 300 else 1


def command_payload(command: str, args: argparse.Namespace) -> dict[str, object]:
    if command in ("stop", "estop"):
        return {"command": "stop"}
    if command == "smoke-move":
        return {"command": "smoke_move"}
    if command == "arms-up":
        return {"command": "arms-up", "amount": args.amount, "ramp": args.ramp, "hold": args.hold}
    if command == "forward":
        return {"command": "forward", "speed": args.speed, "duration": args.duration, "ramp": args.ramp}
    if command == "move":
        return {
            "command": "move",
            "vx": args.vx,
            "vy": args.vy,
            "omega": args.omega,
            "duration": args.duration,
            "ramp": args.ramp,
        }
    raise AssertionError(command)


def send_command(host: str, port: int, payload: dict[str, object]) -> int:
    return print_response(*send_json(bridge_url(host, port, "/cmd"), payload))


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "serve":
        return serve(args)
    if args.command == "shoulder-pitch":
        return shoulder_pitch(args)
    if args.command == "health":
        return print_response(*send_json(bridge_url(args.host, args.port, "/health")))
    if args.command == "probe":
        return print_response(*send_json(bridge_url(args.host, args.port, "/probe_loco")))
    if args.command == "mode":
        return print_response(*send_json(bridge_url(args.host, args.port, "/check_motion_mode")))
    try:
        return send_command(args.host, args.port, command_payload(args.command, args))
    except KeyboardInterrupt:
        print("\nCtrl-C: sending stop")
        return send_command(args.host, args.port, {"command": "stop"})


if __name__ == "__main__":
    raise SystemExit(main())
