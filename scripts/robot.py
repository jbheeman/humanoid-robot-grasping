#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import termios
import tty
from urllib import request
from urllib.error import HTTPError, URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_HOST = "192.168.0.212"
DEFAULT_PORT = 8765
DEFAULT_INTERFACE = "wlan0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Unitree G1 command helper.")
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

    subparsers.add_parser("health")
    subparsers.add_parser("probe")
    subparsers.add_parser("mode")
    subparsers.add_parser("stop")
    subparsers.add_parser("estop")
    subparsers.add_parser("smoke-move")
    subparsers.add_parser("sdk-examples", help="List allowlisted Unitree SDK example actions exposed by the bridge.")

    sdk_example = subparsers.add_parser("sdk-example", help="Run an allowlisted Unitree SDK example action via the bridge.")
    sdk_example.add_argument("example", choices=("g1_loco", "g1_arm_action", "motion_switcher"))
    sdk_example.add_argument("action", nargs="?", default="list")
    sdk_example.add_argument("--speed", type=float, default=0.1)
    sdk_example.add_argument("--duration", type=float, default=0.5)
    sdk_example.add_argument("--smoke", action="store_true")

    vendor_example = subparsers.add_parser(
        "vendor-example",
        help="Run a copied Unitree SDK2 example file verbatim. Run this on the robot.",
    )
    vendor_example.add_argument(
        "example",
        nargs="?",
        choices=("list", "g1_loco", "g1_arm_action", "g1_arm5", "g1_arm7", "g1_low_level", "motion_switcher"),
        default="list",
    )
    vendor_example.add_argument("--interface", default=DEFAULT_INTERFACE)
    vendor_example.add_argument("--no-interface-arg", action="store_true")

    arms = subparsers.add_parser("arms-up", help="Move arms through the HTTP bridge high-level arm command.")
    arms.add_argument("amount", nargs="?", type=float, default=0.15)
    arms.add_argument("--ramp", type=float, default=1.5)
    arms.add_argument("--hold", type=float, default=1.0)

    forward = subparsers.add_parser("forward")
    forward.add_argument("speed", nargs="?", type=float, default=0.05)
    forward.add_argument("--duration", type=float, default=0.4)
    forward.add_argument("--ramp", type=float, default=0.15)

    move = subparsers.add_parser("move")
    move.add_argument("--vx", type=float, default=0.0)
    move.add_argument("--vy", type=float, default=0.0)
    move.add_argument("--omega", "--wz", dest="omega", type=float, default=0.0)
    move.add_argument("--duration", type=float, default=0.4)
    move.add_argument("--ramp", type=float, default=0.15)

    drive = subparsers.add_parser("drive", help="Interactive server-side control. Space/x/Ctrl-C sends stop.")
    drive.add_argument("--speed", type=float, default=0.05)
    drive.add_argument("--turn", type=float, default=0.12)
    drive.add_argument("--duration", type=float, default=0.25)
    drive.add_argument("--ramp", type=float, default=0.08)

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
    command = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
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
    return run_script("unitree_motion_bridge.py", command)


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
    return run_script("g1_arms_forward.py", command)


def vendor_example(args: argparse.Namespace) -> int:
    command = [args.example, "--interface", args.interface]
    if args.no_interface_arg:
        command.append("--no-interface-arg")
    return run_script("unitree_vendor_example.py", command)


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


def read_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def drive(args: argparse.Namespace) -> int:
    print("Interactive drive mode")
    print("w/s: forward/back  a/d: strafe  q/e: turn  space/x: stop  Ctrl-C: stop+exit")
    print("This is a software stop path, not a hardware e-stop.")
    try:
        while True:
            key = read_key().lower()
            payload: dict[str, object] | None = None
            if key == "w":
                payload = {"command": "move", "vx": args.speed, "vy": 0.0, "omega": 0.0}
            elif key == "s":
                payload = {"command": "move", "vx": -args.speed, "vy": 0.0, "omega": 0.0}
            elif key == "a":
                payload = {"command": "move", "vx": 0.0, "vy": args.speed, "omega": 0.0}
            elif key == "d":
                payload = {"command": "move", "vx": 0.0, "vy": -args.speed, "omega": 0.0}
            elif key == "q":
                payload = {"command": "move", "vx": 0.0, "vy": 0.0, "omega": args.turn}
            elif key == "e":
                payload = {"command": "move", "vx": 0.0, "vy": 0.0, "omega": -args.turn}
            elif key in (" ", "x"):
                payload = {"command": "stop"}
            elif key in ("\x03", "\x04"):
                send_command(args.host, args.port, {"command": "stop"})
                return 130
            else:
                continue

            if payload.get("command") == "move":
                payload["duration"] = args.duration
                payload["ramp"] = args.ramp
            status, body = send_json(bridge_url(args.host, args.port, "/cmd"), payload)
            ok = 200 <= status < 300
            print(("ok" if ok else f"error {status}") + f": {payload}")
            if not ok:
                print(body)
    except KeyboardInterrupt:
        print("\nCtrl-C: sending stop")
        send_command(args.host, args.port, {"command": "stop"})
        return 130


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "serve":
        return serve(args)
    if args.command == "shoulder-pitch":
        return shoulder_pitch(args)
    if args.command == "vendor-example":
        return vendor_example(args)

    if args.command == "health":
        return print_response(*send_json(bridge_url(args.host, args.port, "/health")))
    if args.command == "probe":
        return print_response(*send_json(bridge_url(args.host, args.port, "/probe_loco")))
    if args.command == "mode":
        return print_response(*send_json(bridge_url(args.host, args.port, "/check_motion_mode")))
    if args.command == "sdk-examples":
        return print_response(*send_json(bridge_url(args.host, args.port, "/sdk/examples")))
    if args.command == "sdk-example":
        payload = {
            "example": args.example,
            "action": args.action,
            "speed": args.speed,
            "duration": args.duration,
            "smoke": args.smoke,
        }
        return print_response(*send_json(bridge_url(args.host, args.port, "/sdk/example"), payload))
    if args.command == "drive":
        return drive(args)

    try:
        return send_command(args.host, args.port, command_payload(args.command, args))
    except KeyboardInterrupt:
        print("\nCtrl-C: sending stop")
        return send_command(args.host, args.port, {"command": "stop"})


if __name__ == "__main__":
    raise SystemExit(main())
