"""GB10 ROS 2 client for supervised G1 manual arm movement."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request
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
MAX_ROBOT_STEP_RAD = 0.05
MAX_MANUAL_TOTAL_DELTA_RAD = 0.20
MAX_IK_WAYPOINT_DISTANCE_M = 0.01
MAX_IK_WAYPOINT_JOINT_DELTA_RAD = 0.35
MAX_IK_CARTESIAN_OFFSET_M = 0.50
MAX_VISION_TARGET_AGE_MS = 500.0
RIGHT_SHOULDER_POSITION_M = (0.0, -0.18, 0.35)
# The D435 tabletop geometry can place a nominal 25 cm standoff beyond the
# G1's practical right-arm reach. Keep the hand on the pointing ray but cap
# its distance from the shoulder, increasing standoff when necessary.
MAX_POINTING_SHOULDER_DISTANCE_M = 0.40


class RemoteArmError(RuntimeError):
    pass


class _HeartbeatKeeper:
    """Publish the session deadman while CPU-bound IK planning is running."""

    def __init__(self, callback: Any, session_id: str, period_s: float = 0.10) -> None:
        self.callback = callback
        self.session_id = session_id
        self.period_s = period_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="g1-manual-arm-heartbeat",
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.callback(self.session_id)
            except Exception:
                # The main status/response path reports transport failures.
                pass
            self.stop_event.wait(self.period_s)


def _arming_status_summary(status: dict[str, Any]) -> str:
    """Keep arming failures useful without dumping the full 14-joint report."""

    fields = (
        ("state", status.get("state")),
        ("weight", status.get("weight")),
        ("fault", status.get("fault_reason")),
        ("hold", status.get("hold_reason")),
        ("standing", status.get("standing")),
        ("motion_mode", status.get("motion_mode_name")),
        ("motion_mode_verified", status.get("motion_mode_verified")),
        ("ownership_verified", status.get("controller_ownership_verified")),
        ("motor_status_verified", status.get("motor_status_verified")),
        ("motor_state_healthy", status.get("motor_state_healthy")),
        ("robot_state_age_ms", status.get("robot_state_age_ms")),
        ("heartbeat_age_ms", status.get("heartbeat_age_ms")),
    )
    return ", ".join(f"{name}={value!r}" for name, value in fields)


def _arming_failure(
    status: dict[str, Any], session_id: str, *, session_seen: bool
) -> Optional[str]:
    """Describe a terminal arming transition, including after session cleanup."""

    state = str(status.get("state") or "")
    status_session = str(status.get("session_id") or "")
    belongs_to_session = status_session == session_id
    if belongs_to_session and state == "HOLDING":
        return "Robot aborted the arm weight ramp: " + _arming_status_summary(status)
    if session_seen and state in ("FAULT", "DISARMED") and not status_session:
        return "Robot left ARMING before reaching ARMED: " + _arming_status_summary(status)
    return None


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
        self.request_publisher = self.node.create_publisher(ArmManualRequest, REQUEST_TOPIC, qos)
        self.heartbeat_publisher = self.node.create_publisher(String, HEARTBEAT_TOPIC, qos)
        self.target_publishers = {
            "left": self.node.create_publisher(ArmSideTarget, LEFT_TOPIC, qos),
            "right": self.node.create_publisher(ArmSideTarget, RIGHT_TOPIC, qos),
        }
        self.responses: dict[str, object] = {}
        self.status: Optional[dict[str, Any]] = None
        self.node.create_subscription(ArmManualResponse, RESPONSE_TOPIC, self._on_response, qos)
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
        self,
        session_id: str,
        seconds: float,
        stop: threading.Event,
        *,
        trace: Optional[list[dict[str, Any]]] = None,
        phase: str = "",
    ) -> None:
        started = time.monotonic()
        deadline = time.monotonic() + seconds
        while not stop.is_set() and time.monotonic() < deadline:
            self.heartbeat(session_id)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if trace is not None and self.status:
                trace.append(
                    {
                        "elapsed_s": time.monotonic() - started,
                        "phase": phase,
                        "state": self.status.get("state"),
                        "weight": self.status.get("weight"),
                        "fault_reason": self.status.get("fault_reason"),
                        "hold_reason": self.status.get("hold_reason"),
                        "measured_arm_q": self.status.get("measured_arm_q"),
                        "measured_arm_dq": self.status.get("measured_arm_dq"),
                        "commanded_arm_q": self.status.get("commanded_arm_q"),
                    }
                )

    def wait_armed(self, session_id: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        session_seen = False
        last_status: Optional[dict[str, Any]] = None
        while time.monotonic() < deadline:
            self.heartbeat(session_id)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if not self.status:
                continue
            last_status = self.status
            if str(last_status.get("session_id") or "") == session_id:
                session_seen = True
                if last_status.get("state") == "ARMED":
                    return last_status
            failure = _arming_failure(last_status, session_id, session_seen=session_seen)
            if failure:
                raise RemoteArmError(failure)
        detail = (
            "no arm status received" if last_status is None else _arming_status_summary(last_status)
        )
        raise RemoteArmError(f"Timed out waiting for robot arm weight ramp; last status: {detail}")

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
    move.add_argument(
        "--joint-delta",
        action="append",
        default=[],
        metavar="JOINT=RAD",
        help="move several joints together; may be repeated",
    )
    move.add_argument(
        "--arm-deltas",
        nargs=7,
        type=float,
        metavar="RAD",
        help=(
            "seven relative deltas in shoulder-pitch, shoulder-roll, "
            "shoulder-yaw, elbow, wrist-roll, wrist-pitch, wrist-yaw order"
        ),
    )
    move.add_argument("--duration", type=float, default=2.0)
    move.add_argument("--hold", type=float, default=0.5)
    move.add_argument("--no-return", action="store_true")
    move.add_argument("--stay", action="store_true", help="hold until Ctrl-C")
    move.add_argument("--trace-output", type=Path)
    ik = commands.add_parser(
        "ik",
        help="move the right hand by a pelvis-frame Cartesian offset using IK",
    )
    ik.add_argument("--dx", type=float, default=0.0, help="forward offset in metres")
    ik.add_argument("--dy", type=float, default=0.0, help="left offset in metres")
    ik.add_argument("--dz", type=float, default=0.0, help="up offset in metres")
    ik.add_argument("--duration", type=float, default=2.0)
    ik.add_argument("--hold", type=float, default=2.0)
    ik.add_argument("--no-return", action="store_true")
    ik.add_argument("--stay", action="store_true", help="hold until Ctrl-C")
    ik.add_argument("--trace-output", type=Path)
    point = commands.add_parser(
        "point",
        help="continuously point at the vision-localized plushie until Ctrl-C",
    )
    point.add_argument("--server", default="http://127.0.0.1:8000")
    point.add_argument("--standoff", type=float, default=0.25)
    point.add_argument("--max-approach", type=float, default=0.05)
    point.add_argument(
        "--duration",
        type=float,
        default=0.45,
        help="seconds per guarded joint target (default: 0.45)",
    )
    point.add_argument("--hold", type=float, default=3.0)
    point.add_argument(
        "--tracking-step",
        type=float,
        default=0.05,
        help="maximum Cartesian correction per update while --stay is active",
    )
    point.add_argument(
        "--tracking-poll",
        type=float,
        default=0.10,
        help="seconds between target checks while --stay is active",
    )
    point.add_argument(
        "--reacquire-samples",
        type=int,
        default=3,
        help="fresh consistent detections required after target loss",
    )
    point.add_argument("--no-return", action="store_true")
    point.add_argument("--stay", action="store_true", help=argparse.SUPPRESS)
    point.add_argument(
        "--once",
        action="store_false",
        dest="stay",
        help="perform one approach, hold briefly, then release",
    )
    point.set_defaults(stay=True)
    point.add_argument("--trace-output", type=Path)
    commands.add_parser("stop", help="stop/release and clear a latched fault")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise RemoteArmError("--timeout must be finite and > 0")
    if args.command == "enable" and (not math.isfinite(args.seconds) or args.seconds <= 0.0):
        raise RemoteArmError("--seconds must be finite and > 0")
    if args.command in ("move", "ik", "point"):
        if not math.isfinite(args.duration) or not 0.1 <= args.duration <= 10.0:
            raise RemoteArmError("--duration must be between 0.1 and 10 seconds")
        if not math.isfinite(args.hold) or args.hold < 0.0:
            raise RemoteArmError("--hold must be finite and >= 0")
        if args.command != "point" and args.stay and args.no_return:
            raise RemoteArmError("use either --stay or --no-return, not both")
    if args.command == "move":
        selectors = sum(
            (
                bool(args.joint),
                bool(args.joint_delta),
                args.arm_deltas is not None,
            )
        )
        if selectors > 1:
            raise RemoteArmError("use only one of --joint, --joint-delta, or --arm-deltas")
        if args.arm_deltas is not None:
            deltas = [delta for delta in args.arm_deltas if abs(delta) > 1e-9]
            if not deltas:
                raise RemoteArmError("--arm-deltas must request at least one movement")
        else:
            deltas = (
                [_parse_joint_delta(value)[1] for value in args.joint_delta]
                if args.joint_delta
                else [args.delta]
            )
        if any(
            not math.isfinite(delta) or not 0.0 < abs(delta) <= MAX_MANUAL_TOTAL_DELTA_RAD
            for delta in deltas
        ):
            raise RemoteArmError(
                f"each absolute delta must be > 0 and <= {MAX_MANUAL_TOTAL_DELTA_RAD:.2f} rad"
            )
    if args.command == "ik":
        offset = (args.dx, args.dy, args.dz)
        if any(not math.isfinite(value) for value in offset):
            raise RemoteArmError("IK offsets must be finite")
        distance = math.sqrt(sum(value * value for value in offset))
        if not 0.0 < distance <= MAX_IK_CARTESIAN_OFFSET_M:
            raise RemoteArmError(
                f"IK offset norm must be > 0 and <= {MAX_IK_CARTESIAN_OFFSET_M:.2f} m"
            )
    if args.command == "point":
        if not math.isfinite(args.standoff) or not 0.10 <= args.standoff <= 0.50:
            raise RemoteArmError("--standoff must be between 0.10 and 0.50 m")
        if not math.isfinite(args.max_approach) or not 0.01 <= args.max_approach <= 0.30:
            raise RemoteArmError("--max-approach must be between 0.01 and 0.30 m")
        if not math.isfinite(args.tracking_step) or not 0.005 <= args.tracking_step <= 0.05:
            raise RemoteArmError("--tracking-step must be between 0.005 and 0.05 m")
        if not math.isfinite(args.tracking_poll) or not 0.10 <= args.tracking_poll <= 2.0:
            raise RemoteArmError("--tracking-poll must be between 0.10 and 2.0 seconds")
        if not 2 <= args.reacquire_samples <= 10:
            raise RemoteArmError("--reacquire-samples must be between 2 and 10")


def _record_result(report: object) -> None:
    result_path = os.environ.get("G1_RESULT_FILE", "").strip()
    if not result_path:
        return
    path = Path(result_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print(report: object) -> None:
    _record_result(report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _write_motion_trace(path: Path, result: dict[str, Any], samples: list[dict[str, Any]]) -> Path:
    trace_path = path.expanduser()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "schema": "g1-manual-arm-trace-v1",
                "result": result,
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return trace_path


def _fetch_vision_target(server: str, timeout_s: float = 1.0) -> dict[str, Any]:
    """Read one fresh, depth-backed plush localization without mutating state."""

    url = f"{server.rstrip('/')}/arm-tracking"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RemoteArmError(f"could not read vision target from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteArmError("vision server returned a non-object /arm-tracking response")
    if not payload.get("depth_valid"):
        reason = payload.get("rejection_reason") or payload.get("status") or "depth_invalid"
        raise RemoteArmError(f"vision target has no valid registered depth: {reason}")
    try:
        target_age_ms = float(payload["target_age_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteArmError("vision target has no valid target_age_ms") from exc
    if not math.isfinite(target_age_ms) or target_age_ms > MAX_VISION_TARGET_AGE_MS:
        raise RemoteArmError(
            f"vision target is stale: {target_age_ms:.1f} ms "
            f"(maximum {MAX_VISION_TARGET_AGE_MS:.0f} ms)"
        )

    normalized: dict[str, Any] = {
        "target_age_ms": target_age_ms,
        "track_id": payload.get("track_id"),
        "detector_confidence": payload.get("detector_confidence"),
        "prediction_source": payload.get("prediction_source"),
    }
    for field in ("object_xyz_m", "predicted_xyz_m"):
        value = payload.get(field)
        if not isinstance(value, list) or len(value) != 3:
            if field == "object_xyz_m":
                raise RemoteArmError("vision target has no 3D object_xyz_m")
            normalized[field] = None
            continue
        try:
            point = tuple(float(component) for component in value)
        except (TypeError, ValueError) as exc:
            raise RemoteArmError(f"vision target {field} is not numeric") from exc
        if not all(math.isfinite(component) for component in point):
            raise RemoteArmError(f"vision target {field} contains non-finite values")
        normalized[field] = point
    evaluation = payload.get("prediction_evaluation")
    verdict = evaluation.get("verdict") if isinstance(evaluation, dict) else None
    fallback = payload.get("alpha_beta_predicted_xyz_m")
    if verdict == "fallback_better_or_equal" and isinstance(fallback, list) and len(fallback) == 3:
        try:
            fallback_point = tuple(float(component) for component in fallback)
        except (TypeError, ValueError):
            fallback_point = ()
        if len(fallback_point) == 3 and all(math.isfinite(value) for value in fallback_point):
            normalized["predicted_xyz_m"] = fallback_point
            normalized["prediction_source"] = "alpha_beta_fallback_selected"
    return normalized


def _signed_progress(start: float, actual: float, requested_delta: float) -> float:
    direction = 1.0 if requested_delta > 0.0 else -1.0
    return direction * (actual - start)


def _point_hand_target(
    object_xyz: Sequence[float], standoff_m: float
) -> Any:
    import numpy as np

    object_point = np.asarray(object_xyz, dtype=float)
    shoulder = np.asarray(RIGHT_SHOULDER_POSITION_M, dtype=float)
    ray = object_point - shoulder
    distance = float(np.linalg.norm(ray))
    if distance <= standoff_m + 0.02:
        raise RemoteArmError(
            "vision object is too close to form the requested "
            f"{standoff_m:.2f} m pointing standoff"
        )
    radial_distance = min(distance - standoff_m, MAX_POINTING_SHOULDER_DISTANCE_M)
    return shoulder + ray / distance * radial_distance


def _guarded_offsets(delta: float) -> list[float]:
    """Split a visible total move into robot-accepted incremental offsets."""

    steps = max(1, math.ceil(abs(delta) / MAX_ROBOT_STEP_RAD - 1e-9))
    return [delta * index / steps for index in range(1, steps + 1)]


def _guarded_joint_path(
    knots: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    """Interpolate every solved path segment within the robot step contract."""

    if len(knots) < 2:
        return []
    path: list[tuple[float, ...]] = []
    for start_values, target_values in zip(knots, knots[1:]):
        start = tuple(float(value) for value in start_values)
        target = tuple(float(value) for value in target_values)
        if len(start) != len(target) or not start:
            raise ValueError("joint path knots must have equal nonzero dimensions")
        maximum_delta = max(abs(end - begin) for begin, end in zip(start, target))
        steps = max(1, math.ceil(maximum_delta / MAX_ROBOT_STEP_RAD - 1e-9))
        for step in range(1, steps + 1):
            ratio = step / steps
            path.append(
                tuple((1.0 - ratio) * begin + ratio * end for begin, end in zip(start, target))
            )
    return path


def _parse_joint_delta(value: str) -> tuple[str, float]:
    try:
        name, raw_delta = str(value).rsplit("=", 1)
        delta = float(raw_delta)
    except (TypeError, ValueError) as exc:
        raise RemoteArmError("--joint-delta must use JOINT=RAD") from exc
    if not name.strip():
        raise RemoteArmError("--joint-delta must name a joint")
    return name.strip(), delta


def _manual_deltas(args: argparse.Namespace, names: Sequence[str]) -> dict[str, float]:
    if args.arm_deltas is not None:
        return {
            name: float(delta)
            for name, delta in zip(names, args.arm_deltas)
            if abs(float(delta)) > 1e-9
        }
    if not args.joint_delta:
        joint_name = args.joint or f"{args.side}_shoulder_pitch_joint"
        if joint_name not in names:
            raise RemoteArmError(
                f"{joint_name!r} is not a canonical {args.side} arm joint: " + ", ".join(names)
            )
        return {joint_name: float(args.delta)}
    result: dict[str, float] = {}
    for value in args.joint_delta:
        joint_name, delta = _parse_joint_delta(value)
        if joint_name not in names:
            raise RemoteArmError(
                f"{joint_name!r} is not a canonical {args.side} arm joint: " + ", ".join(names)
            )
        if joint_name in result:
            raise RemoteArmError(f"duplicate --joint-delta for {joint_name}")
        result[joint_name] = delta
    return result


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
    heartbeat_keeper: _HeartbeatKeeper | None = None
    motion_trace: list[dict[str, Any]] = []
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

        ik_solver = None
        vision_target = None
        if args.command in ("ik", "point"):
            import numpy as np

            from object_tracking.arm_tracking.ik_solver import (
                G1RightArmIK,
                IKUnavailable,
                default_urdf_path,
            )

            repo_root = Path(__file__).resolve().parents[2]
            try:
                # Model/mesh construction can take longer than the arm
                # heartbeat TTL, so complete it while the bridge is disarmed.
                ik_solver = G1RightArmIK(
                    default_urdf_path(repo_root),
                    position_tolerance_m=0.005,
                    orientation_tolerance_rad=0.5,
                    discontinuity_limit_rad=MAX_IK_WAYPOINT_JOINT_DELTA_RAD,
                    translation_weight=400.0,
                    orientation_weight=0.03,
                )
            except IKUnavailable as exc:
                raise RemoteArmError(str(exc)) from exc

        # Fail before taking arm ownership if the perception pipeline is not
        # currently producing a fresh, depth-backed 3D target.
        if args.command == "point":
            vision_target = _fetch_vision_target(args.server)

        enabled = client.request("enable", timeout_s=args.timeout)
        session_id = str(enabled.get("session_id") or "")
        if not session_id:
            raise RemoteArmError("Robot enabled without returning a session ID")
        armed = client.wait_armed(session_id, args.timeout)
        heartbeat_keeper = _HeartbeatKeeper(client.heartbeat, session_id)
        heartbeat_keeper.start()
        if args.command == "enable":
            client.pump_heartbeat(session_id, args.seconds, stop)
            heartbeat_keeper.close()
            heartbeat_keeper = None
            _print(client.stop_and_wait(session_id, "enable_test_complete", args.timeout))
            session_id = ""
            return 0

        side = args.side if args.command == "move" else "right"
        names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        # Enable latches a complete measured pose before the weight ramp. Use
        # that stable desired pose rather than a later noisy LowState sample,
        # otherwise an exact +0.05 command can appear microscopically larger
        # than the robot-side 0.05-rad step limit.
        latched = armed.get("desired_arm_q")
        if not isinstance(latched, list) or len(latched) != 14:
            raise RemoteArmError("Bridge has no complete latched 14-joint baseline")
        measured_before = armed.get("measured_arm_q")
        if not isinstance(measured_before, list) or len(measured_before) != 14:
            raise RemoteArmError("Bridge has no fresh measured pose before movement")
        offset = 0 if side == "left" else 7
        baseline = [float(value) for value in latched[offset : offset + 7]]
        path_knots: list[tuple[float, ...]] = [tuple(baseline)]
        ik_start_transform = None
        ik_target_transform = None
        point_desired_hand_xyz_m = None
        point_route_kind = None
        point_route_knots = None
        point_stages = 0
        tracking_updates = 0
        target_loss_events = 0
        manual_deltas = None
        if args.command == "move":
            manual_deltas = _manual_deltas(args, names)
            # Operator-facing manual deltas intentionally use the opposite sign
            # from Unitree's canonical SDK joint coordinates. Keep this conversion
            # at the manual boundary so URDF/IK joint angles remain canonical.
            sdk_deltas = {name: -delta for name, delta in manual_deltas.items()}
            manual_target = list(baseline)
            for joint_name, delta in sdk_deltas.items():
                manual_target[names.index(joint_name)] += delta
            path_knots.append(tuple(manual_target))
        else:
            assert ik_solver is not None
            ik_start_transform = ik_solver.forward_kinematics(baseline)
            ik_target_transform = ik_start_transform.copy()
            if args.command == "ik":
                cartesian_offset = np.array([args.dx, args.dy, args.dz])
            if args.command == "ik":
                ik_target_transform[:3, 3] += cartesian_offset
            solution_q = tuple(baseline)
            if args.command == "point":
                used_detour = False
                total_route_knots = 1
                while True:
                    # Re-read vision before every guarded approach stage.  The
                    # first route is therefore not committed to an old plush
                    # location; a moving target shifts the next collision-free
                    # segment immediately.
                    vision_target = _fetch_vision_target(args.server)
                    pointing_xyz = (
                        vision_target["predicted_xyz_m"]
                        if vision_target["predicted_xyz_m"] is not None
                        else vision_target["object_xyz_m"]
                    )
                    desired_hand_xyz = _point_hand_target(pointing_xyz, args.standoff)
                    point_desired_hand_xyz_m = desired_hand_xyz.tolist()
                    ik_target_transform[:3, 3] = desired_hand_xyz
                    current_transform = ik_solver.forward_kinematics(solution_q)
                    stage_offset = desired_hand_xyz - current_transform[:3, 3]
                    remaining_distance = float(np.linalg.norm(stage_offset))
                    if remaining_distance <= 0.008:
                        break
                    if point_stages >= 8:
                        raise RemoteArmError(
                            "pointing route exceeded eight guarded approach stages"
                        )
                    if remaining_distance > args.max_approach:
                        stage_offset *= args.max_approach / remaining_distance
                    stage_target = current_transform.copy()
                    stage_target[:3, 3] += stage_offset
                    route = ik_solver.solve_with_collision_detour(
                        stage_target,
                        solution_q,
                        enforce_orientation=False,
                    )
                    if not route.ok or route.q_path is None:
                        raise RemoteArmError(
                            "IK could not build a collision-free pointing route "
                            f"at stage {point_stages + 1}: {route.reason}, "
                            f"position_error={route.position_error_m:.4f} m"
                        )
                    used_detour = used_detour or len(route.q_path) > 2
                    total_route_knots += len(route.q_path) - 1
                    path_knots.extend(route.q_path[1:])
                    solution_q = route.q_path[-1]
                    point_stages += 1
                point_route_kind = "collision_aware_detour" if used_detour else "direct"
                point_route_knots = total_route_knots
            else:
                waypoint_count = max(
                    1,
                    math.ceil(
                        float(np.linalg.norm(cartesian_offset))
                        / MAX_IK_WAYPOINT_DISTANCE_M
                        - 1e-9
                    ),
                )
                for waypoint_index in range(1, waypoint_count + 1):
                    waypoint = ik_start_transform.copy()
                    waypoint[:3, 3] += cartesian_offset * waypoint_index / waypoint_count
                    result = ik_solver.solve(waypoint, solution_q)
                    if not result.ok or result.q_rad is None:
                        raise RemoteArmError(
                            "IK rejected Cartesian waypoint "
                            f"{waypoint_index}/{waypoint_count}: {result.reason}, "
                            f"position_error={result.position_error_m:.4f} m"
                        )
                    solution_q = result.q_rad
                    path_knots.append(solution_q)
            sdk_deltas = {
                name: float(target - start)
                for name, target, start in zip(names, solution_q, baseline)
                if abs(float(target - start)) > 1e-5
            }
            if not sdk_deltas:
                raise RemoteArmError("IK returned the measured pose without a movement")
        guarded_targets = _guarded_joint_path(path_knots)
        step_count = len(guarded_targets)
        sequence = -1
        for step, target in enumerate(guarded_targets, start=1):
            sequence += 1
            client.publish_target(
                side=side,
                session_id=session_id,
                sequence=sequence,
                positions=target,
                duration_s=args.duration,
            )
            client.wait_sequence(side, sequence, session_id, args.timeout)
            client.pump_heartbeat(
                session_id,
                args.duration,
                stop,
                trace=motion_trace,
                phase=f"outward_{step}",
            )
            status = client.status or {}
            if status.get("state") != "ARMED":
                raise RemoteArmError(
                    "bridge left ARMED during movement: "
                    f"state={status.get('state')}, "
                    f"fault={status.get('fault_reason')}"
                )
        client.pump_heartbeat(session_id, args.hold, stop, trace=motion_trace, phase="outward_hold")
        outward_status = client.status or {}
        if outward_status.get("state") != "ARMED":
            raise RemoteArmError(
                "bridge left ARMED during movement: "
                f"state={outward_status.get('state')}, "
                f"fault={outward_status.get('fault_reason')}"
            )
        measured_after = outward_status.get("measured_arm_q")
        if not isinstance(measured_after, list) or len(measured_after) != 14:
            raise RemoteArmError("Bridge has no fresh measured pose after movement")
        measured_progress: dict[str, float] = {}
        joint_verification: dict[str, dict[str, float | bool]] = {}
        manual_verification_failures: list[str] = []
        for joint_name, sdk_delta in sdk_deltas.items():
            joint_offset = offset + names.index(joint_name)
            progress = _signed_progress(
                float(measured_before[joint_offset]),
                float(measured_after[joint_offset]),
                sdk_delta,
            )
            measured_progress[joint_name] = progress
            required_displacement = max(0.005, abs(sdk_delta) * 0.5)
            passed = progress >= required_displacement
            joint_verification[joint_name] = {
                "requested_sdk_delta_rad": sdk_delta,
                "measured_progress_rad": progress,
                "required_progress_rad": required_displacement,
                "progress_fraction": (progress / abs(sdk_delta) if abs(sdk_delta) > 1e-9 else 0.0),
                "passed": passed,
            }
            if args.command == "move" and not passed:
                manual_verification_failures.append(joint_name)
        ik_position_error_m = None
        ik_measured_xyz_m = None
        ik_achieved_delta_xyz_m = None
        validation_error = None
        if manual_verification_failures:
            validation_error = "manual joint verification failed: " + ", ".join(
                manual_verification_failures
            )
        point_angular_error_deg = None
        if args.command in ("ik", "point"):
            assert ik_solver is not None and ik_target_transform is not None
            measured_right = [float(value) for value in measured_after[-7:]]
            measured_transform = ik_solver.forward_kinematics(measured_right)
            ik_measured_xyz_m = measured_transform[:3, 3].tolist()
            assert ik_start_transform is not None
            ik_achieved_delta_xyz_m = (
                measured_transform[:3, 3] - ik_start_transform[:3, 3]
            ).tolist()
            ik_position_error_m = float(
                np.linalg.norm(measured_transform[:3, 3] - ik_target_transform[:3, 3])
            )
            if ik_position_error_m > 0.0075:
                validation_error = (
                    "measured hand pose did not reach the IK target: "
                    f"position error {ik_position_error_m:.4f} m"
                )
            if args.command == "point":
                assert vision_target is not None
                object_xyz = np.asarray(vision_target["object_xyz_m"], dtype=float)
                shoulder_xyz = np.asarray(RIGHT_SHOULDER_POSITION_M, dtype=float)
                expected_ray = object_xyz - shoulder_xyz
                measured_ray = measured_transform[:3, 3] - shoulder_xyz
                cosine = float(
                    np.dot(expected_ray, measured_ray)
                    / (np.linalg.norm(expected_ray) * np.linalg.norm(measured_ray))
                )
                point_angular_error_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                # Position and ray direction both matter for a point. A hand
                # that merely leaves the rest pose is not a successful point.
                if ik_position_error_m <= 0.025 and point_angular_error_deg <= 5.0:
                    validation_error = None
                else:
                    validation_error = (
                        "measured hand does not point at the vision target: "
                        f"position error {ik_position_error_m:.4f} m, "
                        f"ray error {point_angular_error_deg:.1f} deg"
                    )
        if args.stay and args.command == "point" and not stop.is_set():
            assert ik_solver is not None
            stable_samples = 0
            previous_track_id: object = None
            previous_object_xyz = None
            target_was_lost = False
            while not stop.is_set():
                try:
                    candidate = _fetch_vision_target(args.server)
                except RemoteArmError as exc:
                    if not target_was_lost:
                        target_loss_events += 1
                    target_was_lost = True
                    stable_samples = 0
                    previous_track_id = None
                    previous_object_xyz = None
                    motion_trace.append(
                        {
                            "phase": "vision_hold",
                            "event": "target_lost",
                            "error": str(exc),
                            "state": (client.status or {}).get("state"),
                            "measured_arm_q": (client.status or {}).get("measured_arm_q"),
                            "commanded_arm_q": (client.status or {}).get("commanded_arm_q"),
                        }
                    )
                    client.pump_heartbeat(
                        session_id,
                        args.tracking_poll,
                        stop,
                        trace=motion_trace,
                        phase="vision_hold",
                    )
                    continue

                candidate_object = np.asarray(candidate["object_xyz_m"], dtype=float)
                same_track = candidate.get("track_id") == previous_track_id
                stable_position = (
                    previous_object_xyz is not None
                    and float(np.linalg.norm(candidate_object - previous_object_xyz)) <= 0.08
                )
                stable_samples = stable_samples + 1 if same_track and stable_position else 1
                previous_track_id = candidate.get("track_id")
                previous_object_xyz = candidate_object
                vision_target = candidate
                if stable_samples < args.reacquire_samples:
                    client.pump_heartbeat(
                        session_id,
                        args.tracking_poll,
                        stop,
                        trace=motion_trace,
                        phase="vision_reacquiring",
                    )
                    continue
                if target_was_lost:
                    motion_trace.append(
                        {
                            "phase": "vision_reacquired",
                            "event": "target_reacquired",
                            "track_id": candidate.get("track_id"),
                            "stable_samples": stable_samples,
                        }
                    )
                target_was_lost = False

                pointing_xyz = (
                    candidate["predicted_xyz_m"]
                    if candidate["predicted_xyz_m"] is not None
                    else candidate["object_xyz_m"]
                )
                desired_xyz = _point_hand_target(pointing_xyz, args.standoff)
                current_transform = ik_solver.forward_kinematics(solution_q)
                correction = desired_xyz - current_transform[:3, 3]
                correction_distance = float(np.linalg.norm(correction))
                if correction_distance < 0.005:
                    client.pump_heartbeat(
                        session_id,
                        args.tracking_poll,
                        stop,
                        trace=motion_trace,
                        phase="tracking_deadband_hold",
                    )
                    continue
                if correction_distance > args.tracking_step:
                    correction *= args.tracking_step / correction_distance
                tracking_target = current_transform.copy()
                tracking_target[:3, 3] += correction
                route = ik_solver.solve_with_collision_detour(
                    tracking_target,
                    solution_q,
                    enforce_orientation=False,
                )
                if not route.ok or route.q_path is None:
                    motion_trace.append(
                        {
                            "phase": "tracking_hold",
                            "event": "tracking_route_rejected",
                            "reason": route.reason,
                            "position_error_m": route.position_error_m,
                        }
                    )
                    client.pump_heartbeat(
                        session_id,
                        args.tracking_poll,
                        stop,
                        trace=motion_trace,
                        phase="tracking_hold",
                    )
                    continue
                tracking_targets = _guarded_joint_path(route.q_path)
                for target in tracking_targets:
                    if stop.is_set():
                        break
                    sequence += 1
                    client.publish_target(
                        side=side,
                        session_id=session_id,
                        sequence=sequence,
                        positions=target,
                        duration_s=args.duration,
                    )
                    client.wait_sequence(side, sequence, session_id, args.timeout)
                    client.pump_heartbeat(
                        session_id,
                        args.duration,
                        stop,
                        trace=motion_trace,
                        phase=f"tracking_{tracking_updates}",
                    )
                    status = client.status or {}
                    if status.get("state") != "ARMED":
                        raise RemoteArmError(
                            "bridge left ARMED during tracking: "
                            f"state={status.get('state')}, "
                            f"fault={status.get('fault_reason')}"
                        )
                solution_q = route.q_path[-1]
                tracking_updates += 1
        elif args.stay and not stop.is_set():
            while not stop.is_set():
                client.pump_heartbeat(
                    session_id,
                    0.25,
                    stop,
                    trace=motion_trace,
                    phase="stay",
                )
                status = client.status or {}
                if status.get("state") != "ARMED":
                    raise RemoteArmError(
                        "bridge left ARMED while holding target: "
                        f"state={status.get('state')}, "
                        f"fault={status.get('fault_reason')}"
                    )
        # Pointing is intentionally a terminal pose, not an out-and-back demo.
        # A SIGINT (or the explicit stop command) remains the single release
        # path.  Manual joint and Cartesian test motions retain their optional
        # return behavior.
        if args.command != "point" and not args.no_return and not args.stay and not stop.is_set():
            return_targets = list(reversed([tuple(baseline), *guarded_targets[:-1]]))
            for step, return_target in enumerate(return_targets, start=1):
                sequence += 1
                client.publish_target(
                    side=side,
                    session_id=session_id,
                    sequence=sequence,
                    positions=return_target,
                    duration_s=args.duration,
                )
                client.wait_sequence(side, sequence, session_id, args.timeout)
                client.pump_heartbeat(
                    session_id,
                    args.duration,
                    stop,
                    trace=motion_trace,
                    phase=f"return_{step}",
                )
        heartbeat_keeper.close()
        heartbeat_keeper = None
        final = client.stop_and_wait(session_id, "manual_move_complete", args.timeout)
        session_id = ""
        report = {
            "ok": validation_error is None,
            "error": validation_error,
            "mode": args.command,
            "manual_deltas_rad": manual_deltas,
            "sdk_deltas_rad": sdk_deltas,
            "guarded_steps": step_count,
            "measured_outward_progress_rad": measured_progress,
            "joint_verification": joint_verification,
            "ik_start_xyz_m": (
                None if ik_start_transform is None else ik_start_transform[:3, 3].tolist()
            ),
            "ik_target_xyz_m": (
                None if ik_target_transform is None else ik_target_transform[:3, 3].tolist()
            ),
            "ik_measured_xyz_m": ik_measured_xyz_m,
            "ik_achieved_delta_xyz_m": ik_achieved_delta_xyz_m,
            "ik_position_error_m": ik_position_error_m,
            "vision_object_xyz_m": (
                None if vision_target is None else list(vision_target["object_xyz_m"])
            ),
            "vision_predicted_xyz_m": (
                None
                if vision_target is None or vision_target["predicted_xyz_m"] is None
                else list(vision_target["predicted_xyz_m"])
            ),
            "vision_target_age_ms": (
                None if vision_target is None else vision_target["target_age_ms"]
            ),
            "point_angular_error_deg": point_angular_error_deg,
            "point_standoff_m": (args.standoff if args.command == "point" else None),
            "point_desired_hand_xyz_m": point_desired_hand_xyz_m,
            "point_route_kind": point_route_kind,
            "point_route_knots": point_route_knots,
            "point_stages": point_stages,
            "tracking_updates": tracking_updates,
            "target_loss_events": target_loss_events,
            "vision_track_id": (None if vision_target is None else vision_target["track_id"]),
            "vision_detector_confidence": (
                None if vision_target is None else vision_target["detector_confidence"]
            ),
            "returned": args.command != "point" and not args.no_return and not args.stay,
            "bridge": final,
        }
        if args.trace_output is not None:
            trace_path = _write_motion_trace(args.trace_output, report, motion_trace)
            report["trace_output"] = str(trace_path)
            report["trace_samples"] = len(motion_trace)
        _print(report)
        return 0 if validation_error is None else 2
    except Exception as exc:
        trace_output = getattr(args, "trace_output", None)
        if trace_output is not None:
            _write_motion_trace(
                trace_output,
                {
                    "ok": False,
                    "mode": args.command,
                    "error": f"{type(exc).__name__}: {exc}",
                    "bridge": client.status,
                },
                motion_trace,
            )
        raise
    finally:
        if heartbeat_keeper is not None:
            heartbeat_keeper.close()
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
        _record_result({"ok": False, "error": str(exc)})
        print(f"manual arm command failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
