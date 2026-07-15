"""GB10 ROS 2 client for supervised G1 manual arm movement."""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
import uuid
from typing import Any, Optional, Sequence

from object_tracking.arm_tracking.joints import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)


BASE = "/g1/arm_control"
LEFT_TOPIC = f"{BASE}/left/command"
RIGHT_TOPIC = f"{BASE}/right/command"
HEARTBEAT_TOPIC = f"{BASE}/heartbeat"
REQUEST_TOPIC = f"{BASE}/request"
RESPONSE_TOPIC = f"{BASE}/response"
STATUS_TOPIC = f"{BASE}/status"


class RemoteArmError(RuntimeError):
    pass


class ManualArmClient:
    def __init__(self) -> None:
        try:
            import rclpy
            from g1_control_interfaces.msg import (
                ArmManualRequest,
                ArmManualResponse,
                ArmSideTarget,
            )
            from g1_control_interfaces.msg import ArmState as ArmStateMessage
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String
        except ImportError as exc:
            raise RemoteArmError(
                "ROS 2 interfaces are unavailable. Run through scripts/gb10/arm-remote.sh "
                "after rebuilding the GB10 ROS workspace."
            ) from exc
        self.rclpy = rclpy
        self.ArmManualRequest = ArmManualRequest
        self.ArmSideTarget = ArmSideTarget
        self.String = String
        rclpy.init(args=None)
        self.node = rclpy.create_node("g1_manual_arm_client")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.request_publisher = self.node.create_publisher(
            ArmManualRequest, REQUEST_TOPIC, qos
        )
        self.heartbeat_publisher = self.node.create_publisher(String, HEARTBEAT_TOPIC, qos)
        self.target_publishers = {
            "left": self.node.create_publisher(ArmSideTarget, LEFT_TOPIC, qos),
            "right": self.node.create_publisher(ArmSideTarget, RIGHT_TOPIC, qos),
        }
        self.responses: dict[str, object] = {}
        self.status: Optional[dict[str, Any]] = None
        self.node.create_subscription(
            ArmManualResponse, RESPONSE_TOPIC, self._on_response, qos
        )
        self.node.create_subscription(ArmStateMessage, STATUS_TOPIC, self._on_status, qos)

    def _on_response(self, message: object) -> None:
        self.responses[str(message.request_id)] = message

    def _on_status(self, message: object) -> None:
        try:
            report = json.loads(str(message.report_json))
        except (TypeError, ValueError):
            report = {}
        if isinstance(report, dict):
            self.status = report

    def spin_until(self, predicate: Any, timeout_s: float, description: str) -> Any:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            value = predicate()
            if value:
                return value
        raise RemoteArmError(f"Timed out waiting for {description} after {timeout_s:.1f}s")

    def wait_status(self, timeout_s: float) -> dict[str, Any]:
        return self.spin_until(lambda: self.status, timeout_s, STATUS_TOPIC)

    def request(
        self,
        operation: str,
        *,
        session_id: str = "",
        reason: str = "",
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        message = self.ArmManualRequest()
        message.request_id = request_id
        message.operation = operation
        message.session_id = session_id
        message.reason = reason
        self.request_publisher.publish(message)
        response = self.spin_until(
            lambda: self.responses.pop(request_id, None),
            timeout_s,
            f"{operation} response",
        )
        if not bool(response.ok):
            raise RemoteArmError(
                f"{operation} rejected [{response.error_code}]: {response.message}"
            )
        try:
            report = json.loads(str(response.report_json))
        except (TypeError, ValueError) as exc:
            raise RemoteArmError(f"{operation} returned invalid status JSON") from exc
        if not isinstance(report, dict):
            raise RemoteArmError(f"{operation} returned a non-object status")
        return report

    def heartbeat(self, session_id: str) -> None:
        message = self.String()
        message.data = session_id
        self.heartbeat_publisher.publish(message)

    def publish_target(
        self,
        *,
        side: str,
        session_id: str,
        sequence: int,
        positions: Sequence[float],
        duration_s: float,
    ) -> None:
        message = self.ArmSideTarget()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.session_id = session_id
        message.sequence = int(sequence)
        message.joint_names = list(
            LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        )
        message.position_rad = [float(value) for value in positions]
        seconds = int(duration_s)
        message.move_duration.sec = seconds
        message.move_duration.nanosec = int(round((duration_s - seconds) * 1e9))
        self.target_publishers[side].publish(message)

    def pump_heartbeat(
        self, session_id: str, seconds: float, stop: threading.Event
    ) -> None:
        deadline = time.monotonic() + seconds
        while not stop.is_set() and time.monotonic() < deadline:
            self.heartbeat(session_id)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_armed(self, session_id: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.heartbeat(session_id)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.status and self.status.get("session_id") == session_id:
                if self.status.get("state") == "ARMED":
                    return self.status
                if self.status.get("state") == "FAULT":
                    raise RemoteArmError(f"Robot entered FAULT: {self.status.get('fault_reason')}")
        raise RemoteArmError("Timed out waiting for robot arm weight ramp")

    def stop_and_wait(self, session_id: str, reason: str, timeout_s: float) -> dict[str, Any]:
        self.request("stop", session_id=session_id, reason=reason, timeout_s=timeout_s)
        return self.spin_until(
            lambda: self.status if self.status and self.status.get("state") == "DISARMED" else None,
            timeout_s,
            "zero-weight arm release",
        )

    def wait_sequence(
        self, side: str, sequence: int, session_id: str, timeout_s: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.heartbeat(session_id)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            status = self.status or {}
            rejection = status.get("last_rejection")
            if rejection and rejection.get("side") == side:
                raise RemoteArmError(
                    f"target rejected [{rejection.get('code')}]: {rejection.get('message')}"
                )
            if int(status.get("last_sequences", {}).get(side, -1)) >= sequence:
                return status
        raise RemoteArmError("Timed out waiting for the robot to accept the target")

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=5.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="read bridge status without enabling movement")
    enable = commands.add_parser("enable", help="enable and maintain a heartbeat until stopped")
    enable.add_argument("--seconds", type=float, default=10.0)
    move = commands.add_parser("move", help="move one joint, optionally return, then release")
    move.add_argument("--side", choices=("left", "right"), default="right")
    move.add_argument("--joint", help="canonical joint name (default: shoulder pitch)")
    move.add_argument("--delta", type=float, default=0.05)
    move.add_argument("--duration", type=float, default=2.0)
    move.add_argument("--hold", type=float, default=0.5)
    move.add_argument("--no-return", action="store_true")
    commands.add_parser("stop", help="stop/release and clear a latched fault")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise RemoteArmError("--timeout must be finite and > 0")
    if args.command == "enable" and (not math.isfinite(args.seconds) or args.seconds <= 0.0):
        raise RemoteArmError("--seconds must be finite and > 0")
    if args.command == "move":
        if not math.isfinite(args.delta) or not 0.0 < abs(args.delta) <= 0.05:
            raise RemoteArmError("absolute --delta must be > 0 and <= 0.05 rad")
        if not math.isfinite(args.duration) or not 0.1 <= args.duration <= 10.0:
            raise RemoteArmError("--duration must be between 0.1 and 10 seconds")
        if not math.isfinite(args.hold) or args.hold < 0.0:
            raise RemoteArmError("--hold must be finite and >= 0")


def _print(report: object) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    client = ManualArmClient()
    stop = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    session_id = ""
    try:
        status = client.wait_status(args.timeout)
        if args.command == "inspect":
            _print(
                {
                    "ok": True,
                    "movement_requested": False,
                    "status_topic": STATUS_TOPIC,
                    "left_command_topic": LEFT_TOPIC,
                    "right_command_topic": RIGHT_TOPIC,
                    "bridge": status,
                }
            )
            return 0
        if args.command == "stop":
            _print(client.request("stop", reason="gb10_operator_stop", timeout_s=args.timeout))
            return 0

        enabled = client.request("enable", timeout_s=args.timeout)
        session_id = str(enabled.get("session_id") or "")
        if not session_id:
            raise RemoteArmError("Robot enabled without returning a session ID")
        armed = client.wait_armed(session_id, args.timeout)
        if args.command == "enable":
            client.pump_heartbeat(session_id, args.seconds, stop)
            _print(client.stop_and_wait(session_id, "enable_test_complete", args.timeout))
            session_id = ""
            return 0

        names = LEFT_ARM_JOINT_NAMES if args.side == "left" else RIGHT_ARM_JOINT_NAMES
        joint_name = args.joint or f"{args.side}_shoulder_pitch_joint"
        if joint_name not in names:
            raise RemoteArmError(
                f"{joint_name!r} is not a canonical {args.side} arm joint: {', '.join(names)}"
            )
        # Enable latches a complete measured pose before the weight ramp. Use
        # that stable desired pose rather than a later noisy LowState sample,
        # otherwise an exact +0.05 command can appear microscopically larger
        # than the robot-side 0.05-rad step limit.
        latched = armed.get("desired_arm_q")
        if not isinstance(latched, list) or len(latched) != 14:
            raise RemoteArmError("Bridge has no complete latched 14-joint baseline")
        offset = 0 if args.side == "left" else 7
        baseline = [float(value) for value in latched[offset : offset + 7]]
        target = list(baseline)
        target[names.index(joint_name)] += args.delta
        client.publish_target(
            side=args.side,
            session_id=session_id,
            sequence=0,
            positions=target,
            duration_s=args.duration,
        )
        client.wait_sequence(args.side, 0, session_id, args.timeout)
        client.pump_heartbeat(session_id, args.duration + args.hold, stop)
        if not args.no_return and not stop.is_set():
            client.publish_target(
                side=args.side,
                session_id=session_id,
                sequence=1,
                positions=baseline,
                duration_s=args.duration,
            )
            client.wait_sequence(args.side, 1, session_id, args.timeout)
            client.pump_heartbeat(session_id, args.duration, stop)
        final = client.stop_and_wait(session_id, "manual_move_complete", args.timeout)
        session_id = ""
        _print(
            {
                "ok": True,
                "joint": joint_name,
                "delta_rad": args.delta,
                "returned": not args.no_return,
                "bridge": final,
            }
        )
        return 0
    finally:
        if session_id:
            try:
                client.request("stop", session_id=session_id, reason="client_exit", timeout_s=1.0)
            except Exception:
                pass
        client.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (RemoteArmError, ValueError) as exc:
        print(f"manual arm command failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
