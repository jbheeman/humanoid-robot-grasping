#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from urllib import request
from urllib.error import HTTPError, URLError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send simple HTTP commands to the Unitree motion bridge.")
    parser.add_argument("--host", default="192.168.0.212", help="Robot Wi-Fi IP. Default: 192.168.0.212")
    parser.add_argument("--port", type=int, default=8765)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health")
    subparsers.add_parser("probe")
    subparsers.add_parser("mode")
    subparsers.add_parser("stop")

    arms = subparsers.add_parser("arms-up")
    arms.add_argument("amount", nargs="?", type=float, default=0.15, help="Fraction of forward arm target.")
    arms.add_argument("--ramp", type=float, default=1.5, help="Ramp seconds.")
    arms.add_argument("--hold", type=float, default=1.0, help="Hold seconds.")

    forward = subparsers.add_parser("forward")
    forward.add_argument("speed", nargs="?", type=float, default=0.05, help="Forward speed.")
    forward.add_argument("--duration", type=float, default=0.3)
    forward.add_argument("--ramp", type=float, default=0.15)

    move = subparsers.add_parser("move")
    move.add_argument("--vx", type=float, default=0.0)
    move.add_argument("--vy", type=float, default=0.0)
    move.add_argument("--omega", type=float, default=0.0)
    move.add_argument("--duration", type=float, default=0.3)
    move.add_argument("--ramp", type=float, default=0.15)

    return parser


def bridge_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def send_json(url: str, payload: dict[str, object] | None = None) -> tuple[int, str]:
    if payload is None:
        req = request.Request(url, method="GET")
    else:
        raw = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=raw,
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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "health":
        status, body = send_json(bridge_url(args.host, args.port, "/health"))
    elif args.command == "probe":
        status, body = send_json(bridge_url(args.host, args.port, "/probe_loco"))
    elif args.command == "mode":
        status, body = send_json(bridge_url(args.host, args.port, "/check_motion_mode"))
    elif args.command == "stop":
        status, body = send_json(bridge_url(args.host, args.port, "/cmd"), {"command": "stop"})
    elif args.command == "arms-up":
        status, body = send_json(
            bridge_url(args.host, args.port, "/cmd"),
            {
                "command": "arms-up",
                "amount": args.amount,
                "ramp": args.ramp,
                "hold": args.hold,
            },
        )
    elif args.command == "forward":
        status, body = send_json(
            bridge_url(args.host, args.port, "/cmd"),
            {
                "command": "forward",
                "speed": args.speed,
                "duration": args.duration,
                "ramp": args.ramp,
            },
        )
    elif args.command == "move":
        status, body = send_json(
            bridge_url(args.host, args.port, "/cmd"),
            {
                "command": "move",
                "vx": args.vx,
                "vy": args.vy,
                "omega": args.omega,
                "duration": args.duration,
                "ramp": args.ramp,
            },
        )
    else:
        raise AssertionError(args.command)

    print(body, end="" if body.endswith("\n") else "\n")
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
