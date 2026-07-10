"""Authenticated client for the robot-local G1 arm commissioning service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from .arm_tracking.arm_auth import load_token
from .arm_tracking.arm_commissioning import OPERATOR_ACK
from .arm_tracking.joints import RIGHT_ARM_JOINT_NAMES


def _call(
    base_url: str,
    token: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = None if body is None else json.dumps(body).encode()
    req = request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=raw,
        method="GET" if body is None else "POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=1.0) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read())
        except Exception:
            detail = {"error": "http_error", "message": str(exc)}
        raise SystemExit(json.dumps(detail, indent=2, sort_keys=True)) from exc
    except (URLError, TimeoutError) as exc:
        raise SystemExit(f"commissioning service unavailable: {exc}") from exc


def _session_path(session_id: str, action: str) -> str:
    return f"/api/v1/commissioning/sessions/{session_id}/{action}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commission the G1 right arm without vision or IK")
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--token-file", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    start = subparsers.add_parser("start")
    start.add_argument("--operator", required=True)
    start.add_argument("--client-id", default=socket.gethostname())
    start.add_argument("--ack-cleared-area", action="store_true", required=True)
    for command in ("enable", "heartbeat", "stop", "validate-replay", "promote", "watch"):
        child = subparsers.add_parser(command)
        child.add_argument("session_id")
    jog = subparsers.add_parser("jog")
    jog.add_argument("session_id")
    jog.add_argument("joint_name", choices=RIGHT_ARM_JOINT_NAMES)
    jog.add_argument("direction", choices=("-", "+"))
    jog.add_argument("--sequence", type=int, required=True)
    jog.add_argument("--sign-check", action="store_true")
    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("session_id")
    confirm.add_argument(
        "outcome",
        choices=("expected", "reversed", "no_motion", "unexpected", "accepted", "rejected"),
    )
    confirm.add_argument("--sequence", type=int, required=True)
    confirm.add_argument("--notes", default="")
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("session_id")
    checkpoint.add_argument("label")
    candidate = subparsers.add_parser("capture")
    candidate.add_argument("session_id")
    candidate.add_argument("label", default="safe_chest", nargs="?")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = load_token(token_file=args.token_file)
    if args.command == "status":
        report = _call(args.url, token, "/api/v1/commissioning/state")
    elif args.command == "start":
        report = _call(
            args.url,
            token,
            "/api/v1/commissioning/sessions",
            body={
                "operator_ack": OPERATOR_ACK,
                "operator": args.operator,
                "client_id": args.client_id,
            },
        )
    elif args.command == "jog":
        report = _call(
            args.url,
            token,
            _session_path(args.session_id, "jogs"),
            body={
                "sequence": args.sequence,
                "joint_name": args.joint_name,
                "direction": 1 if args.direction == "+" else -1,
                "kind": "sign_check" if args.sign_check else "jog",
            },
        )
    elif args.command == "confirm":
        report = _call(
            args.url,
            token,
            _session_path(args.session_id, "confirmations"),
            body={"sequence": args.sequence, "outcome": args.outcome, "notes": args.notes},
        )
    elif args.command == "checkpoint":
        report = _call(
            args.url,
            token,
            _session_path(args.session_id, "checkpoints"),
            body={"label": args.label},
        )
    elif args.command == "capture":
        report = _call(
            args.url,
            token,
            _session_path(args.session_id, "candidate"),
            body={"label": args.label},
        )
    elif args.command == "watch":
        try:
            while True:
                report = _call(
                    args.url,
                    token,
                    _session_path(args.session_id, "heartbeat"),
                    body={},
                )
                print(json.dumps(report, indent=2, sort_keys=True))
                time.sleep(0.1)
        except KeyboardInterrupt:
            report = _call(
                args.url,
                token,
                _session_path(args.session_id, "stop"),
                body={"reason": "cli_interrupt"},
            )
    else:
        action = {
            "enable": "enable",
            "heartbeat": "heartbeat",
            "stop": "stop",
            "validate-replay": "replay-validations",
            "promote": "promote",
        }[args.command]
        report = _call(args.url, token, _session_path(args.session_id, action), body={})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
