#!/usr/bin/env python3
"""Robot-local ROS 2 node for guarded arm control and compressed depth.

The Unitree motor topics remain local to the robot.  GB10 exchanges only the
project-level messages and services from ``g1_control_interfaces``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import struct
import threading
from typing import Any

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
    ArmControlMode,
)
from object_tracking.arm_tracking.arm_commissioning import (
    CommissioningConfig,
    CommissioningController,
)
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware
from object_tracking.arm_tracking.calibration import load_calibration
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.manual_arm import ManualArmConfig, ManualArmController
from object_tracking.arm_tracking.joints import ARM_JOINT_NAMES
from object_tracking.ros2_transport import Ros2NodeRunner
from scripts.robot.depth_service import (
    AutoDepthSource,
    DepthService,
    RealSenseDepthSource,
    RealSenseRgbRtpRelay,
    RosAlignedDepthSource,
)


ARM_TARGET_TOPIC = "/g1/arm/target_json"
ARM_STATE_TOPIC = "/g1/arm/state_json"
DEPTH_TOPIC = "/g1/depth"
ARM_CONTROL_REQUEST_TOPIC = "/g1/arm/control/request_json"
ARM_CONTROL_RESPONSE_TOPIC = "/g1/arm/control/response_json"
COMMISSIONING_STATE_TOPIC = "/g1/commissioning/state"
MANUAL_BASE = "/g1/arm_control"
MANUAL_LEFT_TOPIC = f"{MANUAL_BASE}/left/command_json"
MANUAL_RIGHT_TOPIC = f"{MANUAL_BASE}/right/command_json"
MANUAL_HEARTBEAT_TOPIC = f"{MANUAL_BASE}/heartbeat"
MANUAL_REQUEST_TOPIC = f"{MANUAL_BASE}/request_json"
MANUAL_RESPONSE_TOPIC = f"{MANUAL_BASE}/response_json"
MANUAL_STATUS_TOPIC = f"{MANUAL_BASE}/status_json"
MANUAL_JOINT_STATES_TOPIC = f"{MANUAL_BASE}/joint_states"
_HEADER_LENGTH = struct.Struct("!I")
_MAX_MANUAL_WIRE_JSON_BYTES = 16 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded robot-local G1 ROS 2 node")
    parser.add_argument("--calibration")
    parser.add_argument(
        "--control-mode",
        choices=("tracking", "commissioning", "manual"),
        default="tracking",
    )
    parser.add_argument("--allow-movement", action="store_true")
    parser.add_argument(
        "--manual-control-profile",
        choices=("sdk2", "xr"),
        default="sdk2",
        help="Manual arm A/B profile: installed SDK2 example or Unitree XR.",
    )
    parser.add_argument("--expected-motion-mode")
    parser.add_argument(
        "--hardware-interface",
        default="eth0",
        help="Robot NIC used only for native Unitree motor DDS (default: eth0).",
    )
    parser.add_argument("--hardware-domain-id", type=int, default=0)
    parser.add_argument("--robot-id", default="g1")
    parser.add_argument(
        "--commissioning-root",
        type=Path,
        default=Path("runs/research/arm_commissioning"),
    )
    parser.add_argument(
        "--commissioning-profile",
        type=Path,
        default=Path.home() / ".config/g1-grasping/right-arm-home.json",
    )
    parser.add_argument("--depth-source", choices=("auto", "ros", "librealsense"), default="auto")
    parser.add_argument(
        "--ros-image-topic", default="/camera/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument(
        "--ros-camera-info-topic",
        default="/camera/camera/aligned_depth_to_color/camera_info",
    )
    parser.add_argument("--ros-depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-capture-fps", type=int, default=30)
    parser.add_argument("--depth-publish-fps", type=float, default=15.0)
    parser.add_argument(
        "--quiet-healthy-depth",
        action="store_true",
        help="Only print depth health when capture, freshness, or publishing fails.",
    )
    parser.add_argument("--realsense-rgb-target", default="")
    parser.add_argument("--realsense-rgb-port", type=int, default=5600)
    parser.add_argument("--realsense-rgb-fps", type=int, default=60)
    parser.add_argument("--depth-serial")
    parser.add_argument(
        "--disable-depth",
        action="store_true",
        help="Run without opening or publishing depth; intended for arm-only commissioning.",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Publish RealSense depth without constructing any Unitree arm controller.",
    )
    return parser


def _imports() -> dict[str, Any]:
    try:
        from g1_control_interfaces.msg import (
            CommissioningRequest,
            CommissioningResponse,
            CommissioningState,
            CompressedDepth,
        )
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
    except ImportError as exc:  # pragma: no cover - requires sourced ROS workspace
        raise RuntimeError(
            "ROS 2 project interfaces are unavailable; source ros_ws/install/setup.bash"
        ) from exc
    return {
        "CommissioningState": CommissioningState,
        "CommissioningRequest": CommissioningRequest,
        "CommissioningResponse": CommissioningResponse,
        "CompressedDepth": CompressedDepth,
        "JointState": JointState,
        "String": String,
        "QoSProfile": QoSProfile,
        "ReliabilityPolicy": ReliabilityPolicy,
        "DurabilityPolicy": DurabilityPolicy,
        "HistoryPolicy": HistoryPolicy,
    }


def _safe_json(value: object) -> str:
    def finite_json(item: object) -> object:
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {str(key): finite_json(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(nested) for nested in item]
        return item

    return json.dumps(finite_json(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


def _optional_string(value: object) -> str:
    return "" if value is None else str(value)


def _manual_wire_object(message: object) -> dict[str, Any]:
    raw = str(getattr(message, "data", ""))
    if not raw or len(raw.encode("utf-8")) > _MAX_MANUAL_WIRE_JSON_BYTES:
        raise ValueError("manual wire envelope is empty or oversized")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("manual wire envelope must be a JSON object")
    return value


def _age_ms(value: object) -> int:
    if value is None:
        return (1 << 32) - 1
    return max(0, min((1 << 32) - 1, int(round(float(value)))))


class DepthOnlyRosNode:
    """Isolated depth publisher with no native Unitree DDS or arm entities."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.disable_depth:
            raise ValueError("--depth-only cannot be combined with --disable-depth")
        if args.allow_movement:
            raise ValueError("--depth-only never permits movement")
        if args.realsense_rgb_target:
            raise ValueError("--depth-only does not relay RGB")
        if args.depth_publish_fps <= 0.0 or not math.isfinite(args.depth_publish_fps):
            raise ValueError("--depth-publish-fps must be finite and positive")
        self.args = args
        self.types = _imports()
        self.runner = Ros2NodeRunner("g1_robot_depth")
        self.node = self.runner.start()
        depth_qos = self.types["QoSProfile"](
            history=self.types["HistoryPolicy"].KEEP_LAST,
            depth=1,
            reliability=self.types["ReliabilityPolicy"].BEST_EFFORT,
            # Retain exactly one compressed depth frame so a restarted GB10
            # subscriber receives a usable frame immediately after DDS
            # discovery instead of waiting for its next camera period.
            durability=self.types["DurabilityPolicy"].TRANSIENT_LOCAL,
        )
        self.depth_publisher = self.node.create_publisher(
            self.types["CompressedDepth"], DEPTH_TOPIC, depth_qos
        )
        direct = RealSenseDepthSource(
            width=args.depth_width,
            height=args.depth_height,
            fps=args.depth_capture_fps,
            serial=args.depth_serial,
            enable_color=False,
            registered_to_output_rgb=False,
            # The GB10 uses a robust median over the detector ROI.  Avoid
            # running three expensive RealSense filters at 30 Hz on the G1,
            # which otherwise starves the ROS publisher before a frame leaves
            # the robot.
            apply_depth_filters=False,
        )
        if args.depth_source != "librealsense":
            self.runner.close()
            raise ValueError("--depth-only currently requires --depth-source librealsense")
        self.depth_service = DepthService(direct, transmit_fps=args.depth_publish_fps)
        try:
            self.depth_service.start(args.calibration)
        except Exception:
            self.runner.close()
            raise
        self._last_depth_sequence = -1
        self._last_depth_diagnostic = ""
        self._last_publish_error: str | None = None
        self.node.create_timer(1.0 / args.depth_publish_fps, self._publish_depth)
        # Depth capture runs in a background thread, so surface failures here
        # rather than silently leaving a discovered-but-empty ROS topic.
        self.node.create_timer(1.0, self._report_depth_health)

    def _report_depth_health(self) -> None:
        health = self.depth_service.health()
        healthy = (
            bool(health["sensor_fresh"])
            and health["last_error"] is None
            and self._last_publish_error is None
        )
        if self.args.quiet_healthy_depth and healthy:
            return
        status = {
            "event": "depth_status",
            "source": "g1_robot_depth",
            "sequence": health["sequence"],
            "frames_captured": health["frames_captured"],
            "fresh": health["sensor_fresh"],
            "frame_age_ms": health["frame_age_ms"],
            "capture_error": health["last_error"],
            "publish_error": self._last_publish_error,
        }
        encoded = _safe_json(status)
        if encoded != self._last_depth_diagnostic:
            print(encoded, flush=True)
            self._last_depth_diagnostic = encoded

    def _publish_depth(self) -> None:
        with self.depth_service.lock:
            sequence = self.depth_service.latest_sequence
            envelope = self.depth_service.latest_envelope
        if envelope is None or sequence <= self._last_depth_sequence:
            return
        try:
            (header_size,) = _HEADER_LENGTH.unpack(envelope[: _HEADER_LENGTH.size])
            payload_offset = _HEADER_LENGTH.size + header_size
            header = json.loads(envelope[_HEADER_LENGTH.size : payload_offset])
            payload = envelope[payload_offset:]
            message = self.types["CompressedDepth"]()
            message.version = int(header["version"])
            message.sequence = int(header["sequence"])
            message.width = int(header["width"])
            message.height = int(header["height"])
            message.depth_scale = float(header["depth_scale"])
            message.sensor_timestamp_ms = float(header["sensor_timestamp_ms"])
            message.timestamp_domain = str(header["timestamp_domain"])
            message.calibration_id = str(header["calibration_id"])
            message.registered_to_rgb = bool(header.get("registered_to_rgb", False))
            message.encoding = str(header["encoding"])
            message.uncompressed_size = int(header["uncompressed_size"])
            message.checksum_sha256 = str(header["checksum_sha256"])
            message.payload = list(payload)
            self.depth_publisher.publish(message)
            self._last_depth_sequence = sequence
        except Exception as exc:
            self._last_publish_error = f"{type(exc).__name__}: {exc}"
            return

    def close(self) -> None:
        self.depth_service.stop()
        self.runner.close()


class RobotRosNode:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.allow_movement and not args.expected_motion_mode:
            raise ValueError("--allow-movement requires --expected-motion-mode")
        if args.control_mode == "tracking" and args.allow_movement and not args.calibration:
            raise ValueError("tracking movement requires --calibration")
        if args.depth_publish_fps <= 0.0 or not math.isfinite(args.depth_publish_fps):
            raise ValueError("--depth-publish-fps must be finite and positive")

        self.args = args
        self.types = _imports()
        calibration = None if args.calibration is None else load_calibration(args.calibration)
        calibration_id = None if calibration is None else calibration.calibration_id
        self.hardware = UnitreeArmHardware(
            interface=args.hardware_interface,
            domain_id=args.hardware_domain_id,
            expected_motion_mode=args.expected_motion_mode,
        )
        if args.control_mode == "manual":
            control_mode = None
            self.controller = ManualArmController(
                self.hardware,
                ManualArmConfig(
                    allow_movement=args.allow_movement,
                    gain_profile=args.manual_control_profile,
                    control_hz=50.0 if args.manual_control_profile == "sdk2" else 250.0,
                    # XR ownership blending can settle the wrist joints by
                    # roughly 0.04 rad even though the latched target equals
                    # the measured pose.  Use a slower ramp so that benign
                    # arbitration settling remains below the unchanged
                    # measured-velocity safety gate.
                    weight_ramp_s=(
                        0.5 if args.manual_control_profile == "sdk2" else 1.0
                    ),
                    # XR remains under the same target, measured-velocity,
                    # following-error, and collision gates.  This is a modest
                    # increase so visual tracking is not dominated by the
                    # transport waypoint cadence.
                    max_velocity_rad_s=(
                        0.25 if args.manual_control_profile == "sdk2" else 0.60
                    ),
                    max_target_delta_rad=(
                        0.05 if args.manual_control_profile == "sdk2" else 0.10
                    ),
                    max_acceleration_rad_s2=(
                        1.0 if args.manual_control_profile == "sdk2" else 2.00
                    ),
                ),
                event_sink=lambda event: print(
                    _safe_json({"source": "manual_arm_controller", **event}),
                    flush=True,
                ),
            )
        else:
            control_mode = ArmControlMode(args.control_mode)
            config = ArmBridgeConfig(
                control_mode=control_mode,
                allow_movement=args.allow_movement,
                calibration_id=calibration_id,
                joint_contract_id=(
                    joint_contract_id() if control_mode is ArmControlMode.COMMISSIONING else None
                ),
                waist_reference_rad=(
                    None if calibration is None else calibration.waist_reference_rad
                ),
                max_velocity_rad_s=(0.10 if control_mode is ArmControlMode.COMMISSIONING else 0.50),
                max_acceleration_rad_s2=(
                    0.50 if control_mode is ArmControlMode.COMMISSIONING else 2.0
                ),
                max_following_error_rad=(
                    0.05 if control_mode is ArmControlMode.COMMISSIONING else 0.35
                ),
            )
            self.controller = ArmBridgeController(self.hardware, config)
        # SDK2 must create the native Unitree DDS domain before rclpy creates
        # the project ROS domain.  Reversing this order fails on the stock G1
        # image with "ChannelFactory create domain error".
        self.controller.start()
        try:
            self.runner = Ros2NodeRunner("g1_robot_bridge")
            self.node = self.runner.start()
        except Exception:
            self.controller.close()
            raise
        qos = self.types["QoSProfile"](
            history=self.types["HistoryPolicy"].KEEP_LAST,
            depth=5,
            reliability=self.types["ReliabilityPolicy"].RELIABLE,
            durability=self.types["DurabilityPolicy"].VOLATILE,
        )
        depth_qos = self.types["QoSProfile"](
            history=self.types["HistoryPolicy"].KEEP_LAST,
            depth=1,
            reliability=self.types["ReliabilityPolicy"].BEST_EFFORT,
            durability=self.types["DurabilityPolicy"].VOLATILE,
        )
        self.commissioning = (
            CommissioningController(
                self.controller,
                CommissioningConfig(
                    research_root=args.commissioning_root,
                    profile_path=args.commissioning_profile,
                ),
                robot_identity={"robot_id": args.robot_id, "model": "g1-29dof"},
            )
            if control_mode is ArmControlMode.COMMISSIONING
            else None
        )

        self.arm_state_publisher = self.node.create_publisher(
            self.types["String"], ARM_STATE_TOPIC, qos
        )
        self.manual_status_publisher = None
        self.manual_joint_state_publisher = None
        self.manual_response_publisher = None
        self.arm_control_response_publisher = None
        self.commissioning_state_publisher = None
        self.commissioning_response_publisher = None
        self.depth_publisher = None
        if args.control_mode == "manual":
            # Keep the cross-distro command path on standard ROS messages.
            # The stock Foxy Fast DDS reader has intermittently thrown
            # std::bad_alloc while decoding Jazzy-generated custom messages.
            self.manual_status_publisher = self.node.create_publisher(
                self.types["String"], MANUAL_STATUS_TOPIC, depth_qos
            )
            self.manual_joint_state_publisher = self.node.create_publisher(
                self.types["JointState"], MANUAL_JOINT_STATES_TOPIC, depth_qos
            )
            self.manual_response_publisher = self.node.create_publisher(
                self.types["String"], MANUAL_RESPONSE_TOPIC, qos
            )
            self.node.create_subscription(
                self.types["String"],
                MANUAL_LEFT_TOPIC,
                lambda message: self._on_manual_target("left", message),
                qos,
            )
            self.node.create_subscription(
                self.types["String"],
                MANUAL_RIGHT_TOPIC,
                lambda message: self._on_manual_target("right", message),
                qos,
            )
            self.node.create_subscription(
                self.types["String"], MANUAL_HEARTBEAT_TOPIC, self._on_manual_heartbeat, qos
            )
            self.node.create_subscription(
                self.types["String"],
                MANUAL_REQUEST_TOPIC,
                self._on_manual_request,
                qos,
            )
        else:
            self.arm_control_response_publisher = self.node.create_publisher(
                self.types["String"], ARM_CONTROL_RESPONSE_TOPIC, qos
            )
            self.node.create_subscription(
                self.types["String"], ARM_TARGET_TOPIC, self._on_target, qos
            )
            self.node.create_subscription(
                self.types["String"],
                ARM_CONTROL_REQUEST_TOPIC,
                self._on_arm_control_request,
                qos,
            )
        if args.control_mode == "commissioning":
            self.commissioning_state_publisher = self.node.create_publisher(
                self.types["CommissioningState"], COMMISSIONING_STATE_TOPIC, qos
            )
            self.commissioning_response_publisher = self.node.create_publisher(
                self.types["CommissioningResponse"], "/g1/commissioning/response", qos
            )
            self.node.create_subscription(
                self.types["CommissioningRequest"],
                "/g1/commissioning/request",
                self._on_commissioning_request,
                qos,
            )
        self.node.create_timer(0.05, self._publish_state)
        self.node.create_timer(0.05, self._publish_manual_joint_states)
        self.node.create_timer(0.1, self._publish_commissioning_state)

        self.depth_service = None
        self._last_depth_sequence = -1
        if not args.disable_depth:
            self.depth_publisher = self.node.create_publisher(
                self.types["CompressedDepth"], DEPTH_TOPIC, depth_qos
            )
            self.depth_service = self._create_depth_service()
            self.depth_service.start(args.calibration)
            self.node.create_timer(1.0 / args.depth_publish_fps, self._publish_depth)

    def _create_depth_service(self) -> DepthService:
        direct = RealSenseDepthSource(
            width=self.args.depth_width,
            height=self.args.depth_height,
            fps=self.args.depth_capture_fps,
            serial=self.args.depth_serial,
            color_fps=self.args.realsense_rgb_fps,
            enable_color=bool(self.args.realsense_rgb_target),
            registered_to_output_rgb=bool(self.args.realsense_rgb_target),
        )
        ros = RosAlignedDepthSource(
            image_topic=self.args.ros_image_topic,
            camera_info_topic=self.args.ros_camera_info_topic,
            depth_scale=self.args.ros_depth_scale,
        )
        if self.args.depth_source == "ros":
            source = ros
        elif self.args.depth_source == "librealsense":
            source = direct
        else:
            source = AutoDepthSource(ros, direct)
        rgb_relay = None
        if self.args.realsense_rgb_target:
            if self.args.depth_source != "librealsense":
                raise ValueError("--realsense-rgb-target requires --depth-source librealsense")
            rgb_relay = RealSenseRgbRtpRelay(
                host=self.args.realsense_rgb_target,
                port=self.args.realsense_rgb_port,
                fps=self.args.realsense_rgb_fps,
            )
        return DepthService(
            source,
            transmit_fps=self.args.depth_publish_fps,
            rgb_relay=rgb_relay,
        )

    def _on_target(self, message: object) -> None:
        if self.args.control_mode == "manual":
            return
        try:
            payload = _manual_wire_object(message)
            self.controller.set_target(
                session_id=str(payload.get("session_id", "")),
                sequence=int(payload.get("sequence", -1)),
                calibration_id=str(payload.get("calibration_id", "")),
                right_arm_q=tuple(float(value) for value in payload.get("right_arm_q", ())),
                pipeline_age_ms=int(payload.get("pipeline_age_ms", (1 << 32) - 1)),
            )
        except ArmBridgeError:
            return
        except Exception:
            self.controller.stop("ros_target_failure")

    @staticmethod
    def _stamp_ns(stamp: object) -> int:
        return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))

    def _on_manual_target(self, side: str, message: object) -> None:
        if self.args.control_mode != "manual":
            return
        try:
            payload = _manual_wire_object(message)
            self.controller.set_side_target(
                side=side,
                session_id=str(payload.get("session_id", "")),
                sequence=int(payload.get("sequence", -1)),
                joint_names=tuple(payload.get("joint_names", ())),
                position_rad=tuple(payload.get("position_rad", ())),
                duration_s=float(payload.get("duration_s", 0.0)),
                sent_time_ns=int(payload.get("sent_time_ns", 0)),
            )
        except ArmBridgeError as exc:
            # Rejections are reported on the bounded status envelope and do not
            # disturb a currently safe command.
            self.controller.record_rejection(side, exc)
            return
        except Exception as exc:
            self.controller.stop(f"manual_target_failure:{type(exc).__name__}")

    def _on_manual_heartbeat(self, message: object) -> None:
        if self.args.control_mode != "manual":
            return
        try:
            self.controller.heartbeat(str(message.data))
        except ArmBridgeError:
            return

    def _on_manual_request(self, request: object) -> None:
        response = self.types["String"]()
        request_id = ""
        try:
            payload = _manual_wire_object(request)
            request_id = str(payload.get("request_id", ""))
            if self.args.control_mode != "manual":
                raise ArmBridgeError("Manual control mode is not active", code="mode_mismatch")
            operation = str(payload.get("operation", "")).strip().lower()
            if operation == "enable":
                report = self.controller.enable(str(payload.get("session_id", "")))
            elif operation == "heartbeat":
                report = self.controller.heartbeat(str(payload.get("session_id", "")))
            elif operation == "stop":
                report = self.controller.stop(
                    str(payload.get("reason", "") or "operator_stop")
                )
            elif operation == "state":
                report = self.controller.state_report()
            else:
                raise ArmBridgeError("Unknown manual arm operation", code="invalid_request")
        except Exception as exc:
            envelope = {
                "request_id": request_id,
                "ok": False,
                "error_code": str(getattr(exc, "code", "bridge_failure")),
                "message": str(exc),
                "session_id": "",
                "report": {},
            }
        else:
            envelope = {
                "request_id": request_id,
                "ok": True,
                "error_code": "",
                "message": "",
                "session_id": _optional_string(report.get("session_id")),
                "report": report,
            }
        response.data = _safe_json(envelope)
        self.manual_response_publisher.publish(response)

    @staticmethod
    def _service_error(response: object, exc: Exception) -> object:
        response.ok = False
        response.error_code = getattr(exc, "code", "bridge_failure")
        response.message = str(exc)
        response.report_json = "{}"
        return response

    def _on_arm_control_request(self, request: object) -> None:
        response = self.types["String"]()
        request_id = ""
        try:
            payload = _manual_wire_object(request)
            request_id = str(payload.get("request_id", ""))
            operation = str(payload.get("operation", "")).strip().lower()
            if operation == "enable":
                report = self.controller.enable(
                    session_id=str(payload.get("session_id", "")),
                    calibration_id=str(payload.get("calibration_id", "")),
                )
            elif operation == "heartbeat":
                report = self.controller.heartbeat(
                    session_id=str(payload.get("session_id", ""))
                )
            elif operation == "stop":
                report = self.controller.stop(
                    str(payload.get("reason", "") or "operator_stop")
                )
            elif operation == "state":
                report = self.controller.state_report()
            else:
                raise ArmBridgeError("Unknown arm operation", code="invalid_request")
        except Exception as exc:
            envelope = {
                "request_id": request_id,
                "ok": False,
                "error_code": str(getattr(exc, "code", "bridge_failure")),
                "message": str(exc),
                "report": {},
            }
        else:
            envelope = {
                "request_id": request_id,
                "ok": True,
                "error_code": "",
                "message": "",
                "report": report,
            }
        response.data = _safe_json(envelope)
        self.arm_control_response_publisher.publish(response)

    def _run_commissioning_command(
        self, operation_raw: object, request_json: object
    ) -> dict[str, Any]:
        if self.commissioning is None:
            raise ArmBridgeError("Commissioning mode is not active", code="mode_mismatch")
        payload = json.loads(str(request_json or "{}"))
        if not isinstance(payload, dict):
            raise ArmBridgeError("Commissioning payload must be an object")
        operation = str(operation_raw).strip().lower().replace("-", "_")
        session_id = str(payload.get("session_id") or "")
        handlers = {
            "state": lambda: self.commissioning.report(),
            "create_session": lambda: self.commissioning.create_session(
                operator_ack=payload.get("operator_ack"),
                operator=payload.get("operator"),
                client_id=payload.get("client_id"),
            ),
            "enable": lambda: self.commissioning.enable(session_id),
            "heartbeat": lambda: self.commissioning.heartbeat(session_id),
            "jog": lambda: self.commissioning.jog(
                session_id,
                sequence=payload.get("sequence"),
                joint_name=payload.get("joint_name"),
                direction=payload.get("direction"),
                kind=str(payload.get("kind") or "jog"),
            ),
            "confirm": lambda: self.commissioning.confirm_motion(
                session_id,
                sequence=payload.get("sequence"),
                outcome=payload.get("outcome"),
                notes=payload.get("notes", ""),
            ),
            "checkpoint": lambda: self.commissioning.checkpoint(
                session_id, label=payload.get("label")
            ),
            "capture_candidate": lambda: self.commissioning.capture_candidate(
                session_id, label=payload.get("label")
            ),
            "replay_step": lambda: self.commissioning.replay_step(
                session_id, sequence=payload.get("sequence")
            ),
            "validate_replay": lambda: self.commissioning.validate_replay(session_id),
            "stop": lambda: self.commissioning.stop(
                session_id, reason=str(payload.get("reason") or "operator_stop")
            ),
            "promote": lambda: self.commissioning.promote(session_id),
        }
        try:
            return handlers[operation]()
        except KeyError as exc:
            raise ArmBridgeError("Unknown commissioning operation", code="invalid_request") from exc

    def _on_commissioning_request(self, request: object) -> None:
        """Serve GB10 commissioning commands over correlated ROS topics.

        Keep the existing controller semantics and translate only the
        correlated topic transport envelope here.
        """
        response = self.types["CommissioningResponse"]()
        response.request_id = str(getattr(request, "request_id", ""))
        try:
            report = self._run_commissioning_command(
                getattr(request, "operation", ""), getattr(request, "request_json", "{}")
            )
        except Exception as exc:
            response.ok = False
            response.error_code = str(getattr(exc, "code", "bridge_failure"))
            response.message = str(exc)
            response.report_json = "{}"
        else:
            response.ok = True
            response.error_code = ""
            response.message = ""
            response.report_json = _safe_json(report)
        self.commissioning_response_publisher.publish(response)

    def _publish_state(self) -> None:
        report = self.controller.state_report()
        robot_state_age_ms = report.get("robot_state_age_ms")
        report["robot_state_fresh"] = (
            robot_state_age_ms is not None
            and float(robot_state_age_ms) <= self.controller.config.state_ttl_s * 1000.0
        )
        message = self.types["String"]()
        message.data = _safe_json(report)
        self.arm_state_publisher.publish(message)
        if self.args.control_mode == "manual":
            status = self.types["String"]()
            status.data = message.data
            self.manual_status_publisher.publish(status)

    def _publish_manual_joint_states(self) -> None:
        if self.args.control_mode != "manual":
            return
        report = self.controller.state_report()
        measured = report.get("measured_arm_q")
        if measured is None:
            return
        message = self.types["JointState"]()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(ARM_JOINT_NAMES)
        message.position = list(measured)
        velocity = report.get("measured_arm_dq")
        message.velocity = list(velocity if velocity is not None else [])
        self.manual_joint_state_publisher.publish(message)

    def _publish_commissioning_state(self) -> None:
        if self.commissioning is None:
            return
        report = self.commissioning.report()
        message = self.types["CommissioningState"]()
        message.ok = bool(report.get("ok"))
        message.phase = _optional_string(report.get("phase"))
        message.session_id = _optional_string(report.get("session_id"))
        pending = report.get("pending")
        message.has_pending_motion = pending is not None
        message.pending_motion_json = _safe_json(pending) if pending is not None else "{}"
        message.checkpoints_json = [_safe_json(value) for value in report.get("checkpoints", [])]
        message.fault_reason = _optional_string(report.get("fault_reason"))
        message.report_json = _safe_json(report)
        self.commissioning_state_publisher.publish(message)

    def _publish_depth(self) -> None:
        if self.depth_service is None or self.depth_publisher is None:
            return
        with self.depth_service.lock:
            sequence = self.depth_service.latest_sequence
            envelope = self.depth_service.latest_envelope
        if envelope is None or sequence <= self._last_depth_sequence:
            return
        try:
            (header_size,) = _HEADER_LENGTH.unpack(envelope[: _HEADER_LENGTH.size])
            payload_offset = _HEADER_LENGTH.size + header_size
            header = json.loads(envelope[_HEADER_LENGTH.size : payload_offset])
            payload = envelope[payload_offset:]
            message = self.types["CompressedDepth"]()
            message.version = int(header["version"])
            message.sequence = int(header["sequence"])
            message.width = int(header["width"])
            message.height = int(header["height"])
            message.depth_scale = float(header["depth_scale"])
            message.sensor_timestamp_ms = float(header["sensor_timestamp_ms"])
            message.timestamp_domain = str(header["timestamp_domain"])
            message.calibration_id = str(header["calibration_id"])
            message.registered_to_rgb = bool(header.get("registered_to_rgb", False))
            message.encoding = str(header["encoding"])
            message.uncompressed_size = int(header["uncompressed_size"])
            message.checksum_sha256 = str(header["checksum_sha256"])
            message.payload = list(payload)
            self.depth_publisher.publish(message)
            self._last_depth_sequence = sequence
        except Exception:
            return

    def close(self) -> None:
        if self.depth_service is not None:
            self.depth_service.stop()
        self.controller.close()
        self.runner.close()


def main() -> int:
    args = build_parser().parse_args()
    try:
        runtime = DepthOnlyRosNode(args) if args.depth_only else RobotRosNode(args)
    except Exception as exc:
        print(f"robot ROS node startup failed: {exc}")
        return 2
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        _safe_json(
            {
                "ok": True,
                "node": "g1_robot_depth" if args.depth_only else "g1_robot_bridge",
                "allow_movement": args.allow_movement,
                "control_mode": "depth-only" if args.depth_only else args.control_mode,
                "arm_target": ARM_TARGET_TOPIC,
                "arm_state": ARM_STATE_TOPIC,
                "depth": DEPTH_TOPIC,
                "depth_enabled": not args.disable_depth,
            }
        ),
        flush=True,
    )
    try:
        while not stop.wait(0.5):
            pass
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
