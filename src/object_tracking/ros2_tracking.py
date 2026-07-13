"""ROS 2 transport used by the GB10 perception and UI process.

The generated interfaces and ``rclpy`` stay lazily imported so offline tools
and unit tests can import this module without a ROS installation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import struct
import threading
import time
from typing import Any, Callable, Optional, Sequence

from object_tracking.arm_tracking.depth import DepthFrame
from object_tracking.arm_tracking.protocol import DepthEnvelopeCodec, DepthFrameHeader
from object_tracking.ros2_transport import Ros2NodeRunner


ARM_TARGET_TOPIC = "/g1/arm/target"
ARM_STATE_TOPIC = "/g1/arm/state"
DEPTH_TOPIC = "/g1/depth"
ARM_CONTROL_SERVICE = "/g1/arm/control"
COMMISSIONING_COMMAND_SERVICE = "/g1/commissioning/command"
COMMISSIONING_STATE_TOPIC = "/g1/commissioning/state"
_HEADER_LENGTH = struct.Struct("!I")


class RosTrackingError(RuntimeError):
    """The robot-side ROS bridge rejected or could not serve an operation."""

    def __init__(self, message: str, *, code: str = "ros_transport_error") -> None:
        super().__init__(message)
        self.code = code


def _load_types() -> dict[str, Any]:
    try:
        from g1_control_interfaces.msg import (
            ArmState,
            ArmTarget,
            CommissioningState,
            CompressedDepth,
        )
        from g1_control_interfaces.srv import ArmControl, CommissioningCommand
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
    except ImportError as exc:  # pragma: no cover - requires sourced ROS workspace
        raise RosTrackingError(
            "ROS 2 project interfaces are unavailable; source scripts/shared/ros-env.sh"
        ) from exc
    return {
        "ArmState": ArmState,
        "ArmTarget": ArmTarget,
        "CommissioningState": CommissioningState,
        "CompressedDepth": CompressedDepth,
        "ArmControl": ArmControl,
        "CommissioningCommand": CommissioningCommand,
        "QoSProfile": QoSProfile,
        "ReliabilityPolicy": ReliabilityPolicy,
        "DurabilityPolicy": DurabilityPolicy,
        "HistoryPolicy": HistoryPolicy,
    }


class RosTrackingTransport:
    """Thread-safe project ROS client with latest-frame-only depth buffering."""

    def __init__(
        self,
        *,
        runner: Optional[Ros2NodeRunner] = None,
        types: Optional[dict[str, Any]] = None,
        codec: Optional[DepthEnvelopeCodec] = None,
        monotonic: Callable[[], float] = time.monotonic,
        service_timeout_s: float = 2.0,
        state_topic_timeout_s: float = 0.5,
    ) -> None:
        if service_timeout_s <= 0.0:
            raise ValueError("service_timeout_s must be positive")
        if state_topic_timeout_s <= 0.0:
            raise ValueError("state_topic_timeout_s must be positive")
        self._runner = runner
        self._owns_runner = runner is None
        self._types = types
        self._codec = codec or DepthEnvelopeCodec()
        self._monotonic = monotonic
        self._service_timeout_s = service_timeout_s
        self._state_topic_timeout_s = state_topic_timeout_s
        self._node: Optional[object] = None
        self._target_publisher: Optional[object] = None
        self._arm_client: Optional[object] = None
        self._commissioning_client: Optional[object] = None
        self._entities: list[tuple[str, object]] = []
        self._depth_condition = threading.Condition()
        self._latest_depth: Optional[tuple[object, float]] = None
        self._last_depth_sequence = -1
        self._state_lock = threading.Lock()
        self._latest_arm_state: dict[str, Any] = {
            "ok": False,
            "state": "unreachable",
            "reason": "no ROS arm state received",
        }
        self._latest_arm_state_received_at: Optional[float] = None
        self._latest_commissioning_state: dict[str, Any] = {
            "ok": False,
            "phase": "unreachable",
            "reason": "no ROS commissioning state received",
        }
        self._started = False
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("ROS tracking transport is closed")
            types = self._types or _load_types()
            runner = self._runner or Ros2NodeRunner("g1_gb10_tracking")
            node = runner.start()
            qos = types["QoSProfile"](
                history=types["HistoryPolicy"].KEEP_LAST,
                depth=5,
                reliability=types["ReliabilityPolicy"].RELIABLE,
                durability=types["DurabilityPolicy"].VOLATILE,
            )
            depth_qos = types["QoSProfile"](
                history=types["HistoryPolicy"].KEEP_LAST,
                depth=1,
                reliability=types["ReliabilityPolicy"].BEST_EFFORT,
                durability=types["DurabilityPolicy"].VOLATILE,
            )
            try:
                target_publisher = node.create_publisher(
                    types["ArmTarget"], ARM_TARGET_TOPIC, qos
                )
                arm_subscription = node.create_subscription(
                    types["ArmState"], ARM_STATE_TOPIC, self._on_arm_state, qos
                )
                depth_subscription = node.create_subscription(
                    types["CompressedDepth"], DEPTH_TOPIC, self._on_depth, depth_qos
                )
                commissioning_subscription = node.create_subscription(
                    types["CommissioningState"],
                    COMMISSIONING_STATE_TOPIC,
                    self._on_commissioning_state,
                    qos,
                )
                arm_client = node.create_client(types["ArmControl"], ARM_CONTROL_SERVICE)
                commissioning_client = node.create_client(
                    types["CommissioningCommand"], COMMISSIONING_COMMAND_SERVICE
                )
            except Exception:
                if self._owns_runner:
                    runner.close()
                raise
            self._types = types
            self._runner = runner
            self._node = node
            self._target_publisher = target_publisher
            self._arm_client = arm_client
            self._commissioning_client = commissioning_client
            self._entities = [
                ("destroy_subscription", arm_subscription),
                ("destroy_subscription", depth_subscription),
                ("destroy_subscription", commissioning_subscription),
                ("destroy_client", arm_client),
                ("destroy_client", commissioning_client),
                ("destroy_publisher", target_publisher),
            ]
            self._started = True

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            node = self._node
            entities = tuple(self._entities)
            runner = self._runner
            self._entities.clear()
            self._node = None
            self._target_publisher = None
            self._arm_client = None
            self._commissioning_client = None
        with self._depth_condition:
            self._latest_depth = None
            self._depth_condition.notify_all()
        if node is not None:
            for method_name, entity in entities:
                destroy = getattr(node, method_name, None)
                if callable(destroy):
                    destroy(entity)
        if self._owns_runner and runner is not None:
            runner.close()

    def receive_depth(self, timeout_s: float) -> DepthFrame | None:
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative")
        self._require_started()
        deadline = self._monotonic() + timeout_s
        with self._depth_condition:
            while self._latest_depth is None and not self._closed:
                remaining = deadline - self._monotonic()
                if remaining <= 0.0:
                    return None
                self._depth_condition.wait(remaining)
            item = self._latest_depth
            self._latest_depth = None
        if item is None:
            return None
        message, receipt_time = item
        envelope = self._depth_envelope(message)
        decoded = self._codec.decode(envelope)
        header = decoded.header
        if header.sequence <= self._last_depth_sequence:
            return None
        self._last_depth_sequence = header.sequence
        return DepthFrame(
            sequence=header.sequence,
            receipt_time_s=receipt_time,
            z16=decoded.as_numpy().copy(),
            depth_scale=header.depth_scale,
            calibration_id=header.calibration_id,
            sensor_timestamp_ms=header.sensor_timestamp_ms,
            registered_to_rgb=header.registered_to_rgb,
        )

    def arm_state(self) -> dict[str, Any]:
        self._require_started()
        with self._state_lock:
            report = dict(self._latest_arm_state)
            received_at = self._latest_arm_state_received_at
        if received_at is None:
            return report
        age_s = max(0.0, self._monotonic() - received_at)
        if age_s > self._state_topic_timeout_s:
            return {
                "ok": False,
                "state": "unreachable",
                "reason": "ROS arm state topic is stale",
                "last_state": report.get("state"),
                "topic_age_ms": round(age_s * 1000.0, 3),
            }
        report["topic_age_ms"] = round(age_s * 1000.0, 3)
        return report

    def enable_arm(self, session_id: str, calibration_id: str) -> dict[str, Any]:
        if not session_id or not calibration_id:
            raise ValueError("session_id and calibration_id are required")
        return self._arm_call(
            "enable", session_id=session_id, calibration_id=calibration_id
        )

    def heartbeat_arm(self, session_id: str) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required")
        return self._arm_call("heartbeat", session_id=session_id)

    def publish_target(
        self,
        session_id: str,
        sequence: int,
        calibration_id: str,
        right_arm_q: Sequence[float],
        pipeline_age_ms: float,
    ) -> None:
        self._require_started()
        joints = tuple(float(value) for value in right_arm_q)
        if len(joints) != 7:
            raise ValueError("right_arm_q must contain exactly seven joints")
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        message = self._types["ArmTarget"]()
        message.session_id = str(session_id)
        message.sequence = int(sequence)
        message.calibration_id = str(calibration_id)
        message.right_arm_q = list(joints)
        message.pipeline_age_ms = max(0, min((1 << 32) - 1, int(round(pipeline_age_ms))))
        self._target_publisher.publish(message)

    def stop_arm(self, reason: str) -> None:
        self._arm_call("stop", reason=str(reason or "operator_stop"))

    def commissioning(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_started()
        request = self._types["CommissioningCommand"].Request()
        request.operation = str(command)
        request.request_json = json.dumps(
            payload, separators=(",", ":"), sort_keys=True, allow_nan=False
        )
        return self._call_service(self._commissioning_client, request, "commissioning")

    def _arm_call(
        self,
        operation: str,
        *,
        session_id: str = "",
        calibration_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_started()
        request = self._types["ArmControl"].Request()
        request.operation = operation
        request.session_id = session_id
        request.calibration_id = calibration_id
        request.reason = reason
        return self._call_service(self._arm_client, request, f"arm {operation}")

    def _call_service(self, client: object, request: object, operation: str) -> dict[str, Any]:
        available = client.wait_for_service(timeout_sec=self._service_timeout_s)
        if not available:
            raise RosTrackingError(
                f"{operation} service is unavailable after {self._service_timeout_s:.1f}s",
                code="service_unavailable",
            )
        future = client.call_async(request)
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(self._service_timeout_s):
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
            raise RosTrackingError(
                f"{operation} timed out after {self._service_timeout_s:.1f}s",
                code="service_timeout",
            )
        try:
            response = future.result()
        except Exception as exc:
            raise RosTrackingError(f"{operation} failed: {exc}") from exc
        if response is None:
            raise RosTrackingError(f"{operation} returned no response")
        if not bool(response.ok):
            raise RosTrackingError(
                str(response.message or f"{operation} was rejected"),
                code=str(response.error_code or "rejected"),
            )
        try:
            report = json.loads(response.report_json or "{}")
        except json.JSONDecodeError as exc:
            raise RosTrackingError(f"{operation} returned malformed JSON") from exc
        if not isinstance(report, dict):
            raise RosTrackingError(f"{operation} returned a non-object report")
        return report

    def _on_depth(self, message: object) -> None:
        receipt_time = self._monotonic()
        with self._depth_condition:
            self._latest_depth = (message, receipt_time)
            self._depth_condition.notify_all()

    def _on_arm_state(self, message: object) -> None:
        received_at = self._monotonic()
        report = self._message_report(
            message,
            {
                "ok": bool(message.ok),
                "state": str(message.state),
                "session_id": str(message.session_id),
                "control_mode": str(message.control_mode),
                "calibration_id": str(message.calibration_id),
                "weight": float(message.weight),
                "fault_reason": str(message.fault_reason),
                "hold_reason": str(message.hold_reason),
            },
        )
        with self._state_lock:
            self._latest_arm_state = report
            self._latest_arm_state_received_at = received_at

    def _on_commissioning_state(self, message: object) -> None:
        report = self._message_report(
            message,
            {
                "ok": bool(message.ok),
                "phase": str(message.phase),
                "session_id": str(message.session_id),
                "fault_reason": str(message.fault_reason),
            },
        )
        with self._state_lock:
            self._latest_commissioning_state = report

    @staticmethod
    def _message_report(message: object, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            report = json.loads(str(message.report_json or "{}"))
        except (AttributeError, json.JSONDecodeError, TypeError):
            return fallback
        return report if isinstance(report, dict) else fallback

    @staticmethod
    def _depth_envelope(message: object) -> bytes:
        payload = bytes(message.payload)
        header = DepthFrameHeader(
            version=int(message.version),
            sequence=int(message.sequence),
            width=int(message.width),
            height=int(message.height),
            depth_scale=float(message.depth_scale),
            sensor_timestamp_ms=float(message.sensor_timestamp_ms),
            timestamp_domain=str(message.timestamp_domain),
            calibration_id=str(message.calibration_id),
            payload_size=len(payload),
            uncompressed_size=int(message.uncompressed_size),
            checksum_sha256=str(message.checksum_sha256),
            encoding=str(message.encoding),
            byte_order="little",
            registered_to_rgb=bool(message.registered_to_rgb),
        )
        encoded_header = json.dumps(
            asdict(header), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + payload

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("ROS tracking transport is not running")


def create_ros_tracking_transport() -> RosTrackingTransport:
    """Create the production GB10 transport without importing ROS eagerly."""

    return RosTrackingTransport()
