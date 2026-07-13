"""ROS 2 client for the guarded G1 arm commissioning service."""

from __future__ import annotations

import argparse
import json
import socket
import time
from .arm_tracking.joints import RIGHT_ARM_JOINT_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commission the G1 right arm through ROS 2 without vision or IK"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    start = subparsers.add_parser("start")
    start.add_argument("--operator", required=True)
    start.add_argument("--client-id", default=socket.gethostname())
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


def _invoke(args: argparse.Namespace, transport: object) -> dict[str, object]:
    call = transport.commissioning
    if args.command == "status":
        return call("state", {})
    if args.command == "start":
        return call(
            "create_session",
            {
                "operator_ack": True,
                "operator": args.operator,
                "client_id": args.client_id,
            },
        )
    if args.command == "jog":
        return call(
            "jog",
            {
                "session_id": args.session_id,
                "sequence": args.sequence,
                "joint_name": args.joint_name,
                "direction": 1 if args.direction == "+" else -1,
                "kind": "sign_check" if args.sign_check else "jog",
            },
        )
    if args.command == "confirm":
        return call(
            "confirm",
            {
                "session_id": args.session_id,
                "sequence": args.sequence,
                "outcome": args.outcome,
                "notes": args.notes,
            },
        )
    if args.command == "checkpoint":
        return call(
            "checkpoint",
            {"session_id": args.session_id, "label": args.label},
        )
    if args.command == "capture":
        return call(
            "candidate",
            {"session_id": args.session_id, "label": args.label},
        )
    if args.command == "watch":
        try:
            while True:
                report = call("heartbeat", {"session_id": args.session_id})
                print(json.dumps(report, indent=2, sort_keys=True))
                time.sleep(0.1)
        except KeyboardInterrupt:
            return call(
                "stop",
                {"session_id": args.session_id, "reason": "cli_interrupt"},
            )
    operation = {
        "enable": "enable",
        "heartbeat": "heartbeat",
        "stop": "stop",
        "validate-replay": "validate_replay",
        "promote": "promote",
    }[args.command]
    payload: dict[str, object] = {"session_id": args.session_id}
    if operation == "stop":
        payload["reason"] = "operator_stop"
    return call(operation, payload)


def main() -> int:
    args = build_parser().parse_args()
    from .ros2_tracking import create_ros_tracking_transport

    transport = create_ros_tracking_transport()
    try:
        transport.start()
        report = _invoke(args, transport)
    except Exception as exc:
        raise SystemExit(f"ROS 2 commissioning service unavailable: {exc}") from exc
    finally:
        transport.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
