"""ROS 2 transport used by the GB10 perception and UI process.

The generated interfaces and ``rclpy`` stay lazily imported so offline tools
and unit tests can import this module without a ROS installation.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from typing import Any, Callable, Optional, Sequence

from object_tracking.arm_tracking.depth import DepthFrame
from object_tracking.arm_tracking.depth_tcp import DepthTcpReceiver
from object_tracking.arm_tracking.protocol import DepthEnvelopeCodec
from object_tracking.ros2_transport import Ros2NodeRunner


ARM_TARGET_TOPIC = "/g1/arm/target_json"
ARM_STATE_TOPIC = "/g1/arm/state_json"
DEPTH_TOPIC = "/g1/depth_wire"
ARM_CONTROL_REQUEST_TOPIC = "/g1/arm/control/request_json"
ARM_CONTROL_RESPONSE_TOPIC = "/g1/arm/control/response_json"
COMMISSIONING_STATE_TOPIC = "/g1/commissioning/state"
COMMISSIONING_REQUEST_TOPIC = "/g1/commissioning/request"
COMMISSIONING_RESPONSE_TOPIC = "/g1/commissioning/response"


class RosTrackingError(RuntimeError):
    """The robot-side ROS bridge rejected or could not serve an operation."""

    def __init__(self, message: str, *, code: str = "ros_transport_error") -> None:
        super().__init__(message)
        self.code = code


def _load_types() -> dict[str, Any]:
    try:
        from g1_control_interfaces.msg import (
            CommissioningRequest,
            CommissioningResponse,
            CommissioningState,
        )
        from std_msgs.msg import String, UInt8MultiArray
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
        "String": String,
        "CommissioningState": CommissioningState,
        "CommissioningRequest": CommissioningRequest,
        "CommissioningResponse": CommissioningResponse,
        "UInt8MultiArray": UInt8MultiArray,
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
        observe_depth_only: bool = False,
        observe_only: bool = False,
        depth_tcp_receiver: DepthTcpReceiver | None = None,
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
        self._observe_depth_only = bool(observe_depth_only)
        self._observe_only = bool(observe_only)
        self._depth_tcp_receiver = depth_tcp_receiver
        if self._observe_depth_only and self._observe_only:
            raise ValueError("observe_depth_only and observe_only are mutually exclusive")
        self._node: Optional[object] = None
        self._target_publisher: Optional[object] = None
        self._arm_control_publisher: Optional[object] = None
        self._arm_waiters: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._arm_waiters_lock = threading.Lock()
        self._commissioning_request_publisher: Optional[object] = None
        self._commissioning_waiters: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._commissioning_waiters_lock = threading.Lock()
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
            prepare = getattr(runner, "prepare", None)
            node = prepare() if callable(prepare) else runner.start()
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
                durability=types["DurabilityPolicy"].TRANSIENT_LOCAL,
            )
            target_publisher = None
            arm_subscription = None
            commissioning_subscription = None
            commissioning_request_publisher = None
            commissioning_response_subscription = None
            arm_control_publisher = None
            arm_control_subscription = None
            try:
                depth_subscription = node.create_subscription(
                    types["UInt8MultiArray"], DEPTH_TOPIC, self._on_depth, depth_qos
                )
                if self._observe_depth_only:
                    pass
                elif self._observe_only:
                    arm_subscription = node.create_subscription(
                        types["String"], ARM_STATE_TOPIC, self._on_arm_state, qos
                    )
                else:
                    target_publisher = node.create_publisher(
                        types["String"], ARM_TARGET_TOPIC, qos
                    )
                    arm_subscription = node.create_subscription(
                        types["String"], ARM_STATE_TOPIC, self._on_arm_state, qos
                    )
                    commissioning_subscription = node.create_subscription(
                        types["CommissioningState"],
                        COMMISSIONING_STATE_TOPIC,
                        self._on_commissioning_state,
                        qos,
                    )
                    commissioning_request_publisher = node.create_publisher(
                        types["CommissioningRequest"], COMMISSIONING_REQUEST_TOPIC, qos
                    )
                    commissioning_response_subscription = node.create_subscription(
                        types["CommissioningResponse"],
                        COMMISSIONING_RESPONSE_TOPIC,
                        self._on_commissioning_response,
                        qos,
                    )
                    arm_control_publisher = node.create_publisher(
                        types["String"], ARM_CONTROL_REQUEST_TOPIC, qos
                    )
                    arm_control_subscription = node.create_subscription(
                        types["String"],
                        ARM_CONTROL_RESPONSE_TOPIC,
                        self._on_arm_control_response,
                        qos,
                    )
            except Exception:
                if self._owns_runner:
                    runner.close()
                raise
            # Create every endpoint before spinning. Starting the executor
            # first can leave later subscriptions outside its wait set
            # indefinitely on affected rclpy versions.
            runner.start()
            if self._depth_tcp_receiver is not None:
                self._depth_tcp_receiver.start()
            self._types = types
            self._runner = runner
            self._node = node
            self._target_publisher = target_publisher
            self._arm_control_publisher = arm_control_publisher
            self._commissioning_request_publisher = commissioning_request_publisher
            self._entities = [("destroy_subscription", depth_subscription)]
            if self._observe_only:
                self._entities.append(("destroy_subscription", arm_subscription))
            elif not self._observe_depth_only:
                self._entities.extend(
                    [
                        ("destroy_subscription", arm_subscription),
                        ("destroy_subscription", commissioning_subscription),
                        (
                            "destroy_subscription",
                            commissioning_response_subscription,
                        ),
                        ("destroy_subscription", arm_control_subscription),
                        ("destroy_publisher", arm_control_publisher),
                        ("destroy_publisher", commissioning_request_publisher),
                        ("destroy_publisher", target_publisher),
                    ]
                )
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
            self._arm_control_publisher = None
            self._commissioning_request_publisher = None
        with self._depth_condition:
            self._latest_depth = None
            self._depth_condition.notify_all()
        if self._depth_tcp_receiver is not None:
            self._depth_tcp_receiver.close()
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
        if self._depth_tcp_receiver is not None:
            item = self._depth_tcp_receiver.receive(timeout_s)
        else:
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
        payload, receipt_time = item
        envelope = payload if isinstance(payload, bytes) else self._depth_envelope(payload)
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
        return self._arm_call("enable", session_id=session_id, calibration_id=calibration_id)

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
        message = self._types["String"]()
        message.data = json.dumps(
            {
                "session_id": str(session_id),
                "sequence": int(sequence),
                "calibration_id": str(calibration_id),
                "right_arm_q": list(joints),
                "pipeline_age_ms": max(
                    0, min((1 << 32) - 1, int(round(pipeline_age_ms)))
                ),
            },
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        self._target_publisher.publish(message)

    def stop_arm(self, reason: str) -> None:
        self._arm_call("stop", reason=str(reason or "operator_stop"))

    def commissioning(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_started()
        publisher = self._commissioning_request_publisher
        if publisher is None:
            raise RosTrackingError("commissioning request publisher is unavailable")
        request_id = secrets.token_urlsafe(18)
        ready = threading.Event()
        result: dict[str, Any] = {}
        with self._commissioning_waiters_lock:
            self._commissioning_waiters[request_id] = (ready, result)
        request = self._types["CommissioningRequest"]()
        request.request_id = request_id
        request.operation = str(command)
        request.request_json = json.dumps(
            payload, separators=(",", ":"), sort_keys=True, allow_nan=False
        )
        publisher.publish(request)
        if not ready.wait(self._service_timeout_s):
            with self._commissioning_waiters_lock:
                self._commissioning_waiters.pop(request_id, None)
            raise RosTrackingError(
                f"commissioning timed out after {self._service_timeout_s:.1f}s",
                code="service_timeout",
            )
        if not result.get("ok"):
            raise RosTrackingError(
                str(result.get("message") or "commissioning was rejected"),
                code=str(result.get("error_code") or "rejected"),
            )
        report = result.get("report")
        if not isinstance(report, dict):
            raise RosTrackingError("commissioning returned malformed JSON")
        return report

    def _arm_call(
        self,
        operation: str,
        *,
        session_id: str = "",
        calibration_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_started()
        request_id = secrets.token_urlsafe(18)
        ready = threading.Event()
        result: dict[str, Any] = {}
        with self._arm_waiters_lock:
            self._arm_waiters[request_id] = (ready, result)
        request = self._types["String"]()
        request.data = json.dumps(
            {
                "request_id": request_id,
                "operation": operation,
                "session_id": session_id,
                "calibration_id": calibration_id,
                "reason": reason,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self._arm_control_publisher.publish(request)
        if not ready.wait(self._service_timeout_s):
            with self._arm_waiters_lock:
                self._arm_waiters.pop(request_id, None)
            raise RosTrackingError(
                f"arm {operation} timed out after {self._service_timeout_s:.1f}s",
                code="service_timeout",
            )
        if not result.get("ok"):
            raise RosTrackingError(
                str(result.get("message") or f"arm {operation} was rejected"),
                code=str(result.get("error_code") or "rejected"),
            )
        report = result.get("report")
        if not isinstance(report, dict):
            raise RosTrackingError("arm control returned malformed report")
        return report

    def _on_arm_control_response(self, message: object) -> None:
        try:
            payload = json.loads(str(message.data))
            request_id = str(payload.get("request_id", ""))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self._arm_waiters_lock:
            waiter = self._arm_waiters.pop(request_id, None)
        if waiter is None:
            return
        ready, result = waiter
        result.update(payload)
        ready.set()

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
        try:
            report = json.loads(str(message.data))
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(report, dict):
            return
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

    def _on_commissioning_response(self, message: object) -> None:
        request_id = str(getattr(message, "request_id", ""))
        with self._commissioning_waiters_lock:
            waiter = self._commissioning_waiters.pop(request_id, None)
        if waiter is None:
            return
        ready, result = waiter
        try:
            report = json.loads(str(message.report_json or "{}"))
        except (AttributeError, json.JSONDecodeError, TypeError):
            report = None
        result.update(
            ok=bool(getattr(message, "ok", False)),
            error_code=str(getattr(message, "error_code", "")),
            message=str(getattr(message, "message", "")),
            report=report,
        )
        ready.set()

    @staticmethod
    def _message_report(message: object, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            report = json.loads(str(message.report_json or "{}"))
        except (AttributeError, json.JSONDecodeError, TypeError):
            return fallback
        return report if isinstance(report, dict) else fallback

    @staticmethod
    def _depth_envelope(message: object) -> bytes:
        return bytes(message.data)

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("ROS tracking transport is not running")


def create_ros_tracking_transport(
    *, observe_depth_only: bool = False, observe_only: bool = False
) -> RosTrackingTransport:
    """Create the production GB10 transport without importing ROS eagerly."""

    port = int(os.environ.get("G1_DEPTH_TCP_PORT", "5601"))
    return RosTrackingTransport(
        observe_depth_only=observe_depth_only,
        observe_only=observe_only,
        depth_tcp_receiver=DepthTcpReceiver(port),
    )
