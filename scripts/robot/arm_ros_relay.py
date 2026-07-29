#!/usr/bin/env python3
"""Foxy ROS relay for the guarded arm protocol; contains no Unitree SDK code."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import threading
import uuid
from typing import Any

from object_tracking.ros2_transport import Ros2NodeRunner


ARM_TARGET_TOPIC = "/g1/arm/target_json"
ARM_STATE_TOPIC = "/g1/arm/state_json"
ARM_CONTROL_REQUEST_TOPIC = "/g1/arm/control/request_json"
ARM_CONTROL_RESPONSE_TOPIC = "/g1/arm/control/response_json"
MAX_WIRE_BYTES = 16 * 1024


def _safe_json(value: object) -> str:
    def finite(item: object) -> object:
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {str(key): finite(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(nested) for nested in item]
        return item

    return json.dumps(finite(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


def _wire_object(message: object) -> dict[str, Any]:
    raw = str(getattr(message, "data", ""))
    if not raw or len(raw.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("ROS JSON envelope is empty or oversized")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("ROS JSON envelope must be an object")
    return value


class NativeArmClient:
    def __init__(self, socket_path: str, timeout_s: float = 0.2) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_s

    def call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        encoded = (_safe_json({
            "request_id": request_id,
            "operation": operation,
            "payload": payload,
        }) + "\n").encode("utf-8")
        if len(encoded) > MAX_WIRE_BYTES:
            raise ValueError("Native worker request is oversized")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_s)
            connection.connect(self.socket_path)
            connection.sendall(encoded)
            chunks = bytearray()
            while not chunks.endswith(b"\n"):
                part = connection.recv(min(4096, MAX_WIRE_BYTES + 1 - len(chunks)))
                if not part:
                    raise RuntimeError("Native arm worker closed without a response")
                chunks.extend(part)
                if len(chunks) > MAX_WIRE_BYTES:
                    raise RuntimeError("Native arm worker response is oversized")
        response = json.loads(chunks.decode("utf-8"))
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise RuntimeError("Native arm worker returned an invalid response")
        return response


class ArmRosRelay:
    def __init__(
        self,
        socket_path: str,
        *,
        runner: Ros2NodeRunner | None = None,
        client: NativeArmClient | None = None,
    ) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.String = String
        self.client = client or NativeArmClient(socket_path)
        self._last_reported_state: str | None = None
        self.runner = runner or Ros2NodeRunner("g1_arm_ros_relay")
        # Foxy executors may never add entities created after spin() begins to
        # their wait set. Build the complete relay graph before starting the
        # executor, matching the GB10 transport's proven initialization order.
        self.node = self.runner.prepare()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.state_publisher = self.node.create_publisher(String, ARM_STATE_TOPIC, qos)
        self.response_publisher = self.node.create_publisher(
            String, ARM_CONTROL_RESPONSE_TOPIC, qos
        )
        self.node.create_subscription(String, ARM_TARGET_TOPIC, self._on_target, qos)
        self.node.create_subscription(
            String, ARM_CONTROL_REQUEST_TOPIC, self._on_control_request, qos
        )
        self.node.create_timer(0.05, self._publish_state)
        self.runner.start()

    def _call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.call(operation, payload)

    def _on_target(self, message: object) -> None:
        try:
            response = self._call("target", _wire_object(message))
            if not response.get("ok"):
                print(
                    _safe_json(
                        {
                            "event": "arm_target_rejected",
                            "error_code": response.get("error_code"),
                            "message": response.get("message"),
                        }
                    ),
                    flush=True,
                )
                return
        except Exception as exc:
            print(
                _safe_json(
                    {
                        "event": "arm_target_transport_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            try:
                self._call("stop", {"reason": "ros_target_failure"})
            except Exception:
                pass

    def _on_control_request(self, message: object) -> None:
        request_id = ""
        try:
            payload = _wire_object(message)
            request_id = str(payload.get("request_id", ""))
            operation = str(payload.get("operation", "")).strip().lower()
            if operation not in {"enable", "heartbeat", "return", "stop", "state"}:
                raise ValueError("Unknown arm operation")
            response = self._call(operation, payload)
        except Exception as exc:
            response = {
                "ok": False,
                "error_code": "relay_failure",
                "message": str(exc),
                "report": {},
            }
        envelope = {
            "request_id": request_id,
            "ok": bool(response.get("ok")),
            "error_code": str(response.get("error_code", "")),
            "message": str(response.get("message", "")),
            "report": response.get("report", {}),
        }
        outgoing = self.String()
        outgoing.data = _safe_json(envelope)
        self.response_publisher.publish(outgoing)

    def _publish_state(self) -> None:
        try:
            response = self._call("state", {})
            if not response.get("ok"):
                raise RuntimeError(str(response.get("message") or "worker state failed"))
            report = response.get("report", {})
            state_age_ms = report.get("robot_state_age_ms")
            report["robot_state_fresh"] = (
                state_age_ms is not None and float(state_age_ms) <= 250.0
            )
        except Exception as exc:
            report = {
                "state": "FAULT",
                "fault_reason": "native_worker_unavailable",
                "native_worker_error": f"{type(exc).__name__}: {exc}",
                "robot_state_fresh": False,
            }
        current_state = str(report.get("state") or "UNKNOWN")
        if current_state != self._last_reported_state:
            print(
                _safe_json(
                    {
                        "event": "arm_state_transition",
                        "from": self._last_reported_state,
                        "to": current_state,
                        "weight": report.get("weight"),
                        "hold_reason": report.get("hold_reason"),
                        "fault_reason": report.get("fault_reason"),
                        "fault_details": report.get("fault_details"),
                    }
                ),
                flush=True,
            )
            self._last_reported_state = current_state
        outgoing = self.String()
        outgoing.data = _safe_json(report)
        self.state_publisher.publish(outgoing)

    def close(self) -> None:
        try:
            self._call("stop", {"reason": "ros_relay_shutdown"})
        except Exception:
            pass
        self.runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Foxy ROS to native-arm relay")
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    relay = ArmRosRelay(args.socket)
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(_safe_json({"ok": True, "node": "g1_arm_ros_relay", "socket": args.socket}), flush=True)
    try:
        while not stop.wait(0.5):
            pass
    finally:
        relay.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
