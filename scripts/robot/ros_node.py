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
from object_tracking.ros2_transport import Ros2NodeRunner
from scripts.robot.depth_service import (
    AutoDepthSource,
    DepthService,
    RealSenseDepthSource,
    RosAlignedDepthSource,
)


ARM_TARGET_TOPIC = "/g1/arm/target"
ARM_STATE_TOPIC = "/g1/arm/state"
DEPTH_TOPIC = "/g1/depth"
ARM_CONTROL_SERVICE = "/g1/arm/control"
COMMISSIONING_COMMAND_SERVICE = "/g1/commissioning/command"
COMMISSIONING_STATE_TOPIC = "/g1/commissioning/state"
_HEADER_LENGTH = struct.Struct("!I")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded robot-local G1 ROS 2 node")
    parser.add_argument("--calibration")
    parser.add_argument(
        "--control-mode",
        choices=("tracking", "commissioning"),
        default="tracking",
    )
    parser.add_argument("--allow-movement", action="store_true")
    parser.add_argument("--expected-motion-mode")
    parser.add_argument(
        "--hardware-interface",
        default="wlan0",
        help="Robot NIC used only for native Unitree motor DDS (default: wlan0).",
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
    parser.add_argument(
        "--depth-source", choices=("auto", "ros", "librealsense"), default="auto"
    )
    parser.add_argument("--ros-image-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument(
        "--ros-camera-info-topic",
        default="/camera/camera/aligned_depth_to_color/camera_info",
    )
    parser.add_argument("--ros-depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-capture-fps", type=int, default=30)
    parser.add_argument("--depth-publish-fps", type=float, default=15.0)
    parser.add_argument("--depth-serial")
    parser.add_argument(
        "--disable-depth",
        action="store_true",
        help="Run without opening or publishing depth; intended for arm-only commissioning.",
    )
    return parser


def _imports() -> dict[str, Any]:
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
        raise RuntimeError(
            "ROS 2 project interfaces are unavailable; source ros_ws/install/setup.bash"
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


def _safe_json(value: object) -> str:
    def finite_json(item: object) -> object:
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {str(key): finite_json(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(nested) for nested in item]
        return item

    return json.dumps(
        finite_json(value), separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _optional_string(value: object) -> str:
    return "" if value is None else str(value)


def _age_ms(value: object) -> int:
    if value is None:
        return (1 << 32) - 1
    return max(0, min((1 << 32) - 1, int(round(float(value)))))


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
        self.runner = Ros2NodeRunner("g1_robot_bridge")
        self.node = self.runner.start()
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

        calibration = None if args.calibration is None else load_calibration(args.calibration)
        calibration_id = None if calibration is None else calibration.calibration_id
        control_mode = ArmControlMode(args.control_mode)
        config = ArmBridgeConfig(
            control_mode=control_mode,
            allow_movement=args.allow_movement,
            calibration_id=calibration_id,
            joint_contract_id=(
                joint_contract_id() if control_mode is ArmControlMode.COMMISSIONING else None
            ),
            waist_reference_rad=(None if calibration is None else calibration.waist_reference_rad),
            max_velocity_rad_s=(0.10 if control_mode is ArmControlMode.COMMISSIONING else 0.50),
            max_acceleration_rad_s2=(
                0.50 if control_mode is ArmControlMode.COMMISSIONING else 2.0
            ),
            max_following_error_rad=(
                0.05 if control_mode is ArmControlMode.COMMISSIONING else 0.35
            ),
        )
        self.hardware = UnitreeArmHardware(
            interface=args.hardware_interface,
            domain_id=args.hardware_domain_id,
            expected_motion_mode=args.expected_motion_mode,
        )
        self.controller = ArmBridgeController(self.hardware, config)
        self.controller.start()
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
            self.types["ArmState"], ARM_STATE_TOPIC, qos
        )
        self.commissioning_state_publisher = self.node.create_publisher(
            self.types["CommissioningState"], COMMISSIONING_STATE_TOPIC, qos
        )
        self.depth_publisher = None
        self.node.create_subscription(
            self.types["ArmTarget"], ARM_TARGET_TOPIC, self._on_target, qos
        )
        self.node.create_service(
            self.types["ArmControl"], ARM_CONTROL_SERVICE, self._on_arm_control
        )
        self.node.create_service(
            self.types["CommissioningCommand"],
            COMMISSIONING_COMMAND_SERVICE,
            self._on_commissioning_command,
        )
        self.node.create_timer(0.05, self._publish_state)
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
        return DepthService(source, transmit_fps=self.args.depth_publish_fps)

    def _on_target(self, message: object) -> None:
        try:
            self.controller.set_target(
                session_id=message.session_id,
                sequence=int(message.sequence),
                calibration_id=message.calibration_id,
                right_arm_q=tuple(float(value) for value in message.right_arm_q),
                pipeline_age_ms=int(message.pipeline_age_ms),
            )
        except ArmBridgeError:
            return
        except Exception:
            self.controller.stop("ros_target_failure")

    @staticmethod
    def _service_error(response: object, exc: Exception) -> object:
        response.ok = False
        response.error_code = getattr(exc, "code", "bridge_failure")
        response.message = str(exc)
        response.report_json = "{}"
        return response

    def _on_arm_control(self, request: object, response: object) -> object:
        try:
            operation = str(request.operation).strip().lower()
            if operation == "enable":
                report = self.controller.enable(
                    session_id=request.session_id,
                    calibration_id=request.calibration_id,
                )
            elif operation == "heartbeat":
                report = self.controller.heartbeat(session_id=request.session_id)
            elif operation == "stop":
                report = self.controller.stop(str(request.reason or "operator_stop"))
            elif operation == "state":
                report = self.controller.state_report()
            else:
                raise ArmBridgeError("Unknown arm operation", code="invalid_request")
        except Exception as exc:
            return self._service_error(response, exc)
        response.ok = True
        response.error_code = ""
        response.message = ""
        response.report_json = _safe_json(report)
        return response

    def _on_commissioning_command(self, request: object, response: object) -> object:
        if self.commissioning is None:
            return self._service_error(
                response,
                ArmBridgeError("Commissioning mode is not active", code="mode_mismatch"),
            )
        try:
            payload = json.loads(request.request_json or "{}")
            if not isinstance(payload, dict):
                raise ArmBridgeError("Commissioning payload must be an object")
            operation = str(request.operation).strip().lower().replace("-", "_")
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
                report = handlers[operation]()
            except KeyError as exc:
                raise ArmBridgeError(
                    "Unknown commissioning operation", code="invalid_request"
                ) from exc
        except Exception as exc:
            return self._service_error(response, exc)
        response.ok = True
        response.error_code = ""
        response.message = ""
        response.report_json = _safe_json(report)
        return response

    def _publish_state(self) -> None:
        report = self.controller.state_report()
        message = self.types["ArmState"]()
        message.ok = bool(report.get("ok"))
        message.state = _optional_string(report.get("state"))
        message.session_id = _optional_string(report.get("session_id"))
        message.control_mode = _optional_string(report.get("control_mode"))
        message.calibration_id = _optional_string(report.get("calibration_id"))
        message.last_sequence = int(report.get("last_sequence", -1))
        message.last_target_age_ms = _age_ms(report.get("last_target_age_ms"))
        message.robot_state_age_ms = _age_ms(report.get("robot_state_age_ms"))
        measured = report.get("measured_arm_q")
        commanded = report.get("commanded_arm_q")
        message.measured_arm_q = list(measured if measured is not None else [math.nan] * 14)
        message.commanded_arm_q = list(commanded if commanded is not None else [math.nan] * 14)
        for field in (
            "standing",
            "compatible_motion_mode",
            "controller_available",
            "motion_mode_verified",
            "controller_ownership_verified",
            "motor_status_verified",
            "motor_state_healthy",
        ):
            setattr(message, field, bool(report.get(field)))
        robot_state_age_ms = report.get("robot_state_age_ms")
        message.robot_state_fresh = (
            robot_state_age_ms is not None
            and float(robot_state_age_ms) <= self.controller.config.state_ttl_s * 1000.0
        )
        message.weight = float(report.get("weight", 0.0))
        message.motor_faults = [str(value) for value in report.get("motor_faults", [])]
        message.fault_reason = _optional_string(report.get("fault_reason"))
        message.hold_reason = _optional_string(report.get("hold_reason"))
        message.report_json = _safe_json(report)
        self.arm_state_publisher.publish(message)

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
        message.checkpoints_json = [
            _safe_json(value) for value in report.get("checkpoints", [])
        ]
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
        runtime = RobotRosNode(args)
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
                "node": "g1_robot_bridge",
                "allow_movement": args.allow_movement,
                "control_mode": args.control_mode,
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
