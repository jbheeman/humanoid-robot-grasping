"""Unitree G1 camera helpers and ROS 2 locomotion clients.

The ROS imports are deliberately deferred until a client is started.  Vision
tools can therefore use the camera helpers on hosts without a ROS install.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Callable, Mapping

from object_tracking.ros2_transport import (
    Ros2NodeRunner,
    Ros2RequestResponseClient,
    Ros2RpcResult,
    Ros2RpcTimeout,
    Ros2Unavailable,
)


def default_camera_env() -> list[str]:
    value = os.environ.get("G1_CAMERA_SOURCES")
    if value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def default_camera_ports() -> list[str]:
    value = os.environ.get("G1_CAMERA_UDP_PORTS", "1720")
    ports = [part.strip() for part in value.split(",") if part.strip()]
    return ports or ["1720"]


def default_gstreamer_port_templates() -> list[str]:
    multicast_group = os.environ.get("G1_CAMERA_MULTICAST_GROUP", "230.1.1.1").strip()
    multicast_iface = os.environ.get("G1_CAMERA_MULTICAST_IFACE", "").strip()
    iface = f" multicast-iface={multicast_iface}" if multicast_iface else ""
    return [
        f"udpsrc multicast-group={multicast_group} address=0.0.0.0 port={{port}} "
        f"auto-multicast=true{iface} ! application/x-rtp,media=(string)video,"
        "encoding-name=(string)H264,clock-rate=(int)90000 ! queue ! rtph264depay ! "
        "h264parse ! avdec_h264 ! videoconvert ! appsink",
        f"udpsrc address={multicast_group} port={{port}} auto-multicast=true ! "
        "application/x-rtp,media=(string)video,encoding-name=(string)H264,"
        "clock-rate=(int)90000 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! appsink",
    ]


def default_camera_sources() -> list[str]:
    transport = os.environ.get("G1_CAMERA_TRANSPORT", "gstreamer").lower()
    if transport == "rtsp":
        return [
            "rtsp://{ip}:8554/unicast",
            "rtsp://{ip}:8554/live",
            "rtsp://{ip}:8554/stream",
            "rtsp://{ip}:554/live",
            "rtsp://{ip}:554/stream1",
            "rtsp://{ip}:554/h264Preview_01_main",
        ]

    return [
        template.format(port=port, ip="{ip}")
        for template in default_gstreamer_port_templates()
        for port in default_camera_ports()
    ]


def is_gstreamer_pipeline(source: str) -> bool:
    return isinstance(source, str) and "!" in source and (
        "udpsrc" in source or "gst-launch" in source
    )


DEFAULT_G1_CAMERA_TEMPLATES = tuple(default_camera_sources())


def g1_camera_candidates(robot_ip: str, camera_url: str | None = None) -> list[str]:
    explicit_url = camera_url or os.environ.get("G1_CAMERA_URL")
    if explicit_url:
        return [explicit_url.format(ip=robot_ip)]

    configured = default_camera_env()
    if configured:
        return [candidate.format(ip=robot_ip) for candidate in configured]
    return [template.format(ip=robot_ip) for template in DEFAULT_G1_CAMERA_TEMPLATES]


class UnitreeG1Error(RuntimeError):
    """A rejected, unavailable, or timed-out Unitree robot operation."""


class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26


G1_ARM_FORWARD_TARGETS = {
    G1JointIndex.LeftShoulderPitch: 0.85,
    G1JointIndex.LeftShoulderRoll: 0.20,
    G1JointIndex.LeftShoulderYaw: 0.0,
    G1JointIndex.LeftElbow: 0.20,
    G1JointIndex.LeftWristRoll: 0.0,
    G1JointIndex.RightShoulderPitch: 0.85,
    G1JointIndex.RightShoulderRoll: -0.20,
    G1JointIndex.RightShoulderYaw: 0.0,
    G1JointIndex.RightElbow: 0.20,
    G1JointIndex.RightWristRoll: 0.0,
}


def smoothstep(value: float) -> float:
    clamped = min(max(value, 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


@dataclass(frozen=True)
class UnitreeCommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


LOCO_SERVICE_CHOICES = ("auto", "sport", "ai_sport")
DEFAULT_G1_LOCO_SERVICE_NAME = "sport"

LOCO_API_GET_FSM_ID = 7001
LOCO_API_GET_FSM_MODE = 7002
LOCO_API_GET_BALANCE_MODE = 7003
LOCO_API_GET_SWING_HEIGHT = 7004
LOCO_API_GET_STAND_HEIGHT = 7005
LOCO_API_SET_FSM_ID = 7101
LOCO_API_SET_BALANCE_MODE = 7102
LOCO_API_SET_VELOCITY = 7105

MOTION_API_CHECK_MODE = 1001
MOTION_API_SELECT_MODE = 1002
MOTION_API_RELEASE_MODE = 1003

MAX_FORWARD_SPEED_MPS = 0.5
MAX_LATERAL_SPEED_MPS = 0.3
MAX_YAW_SPEED_RAD_S = 0.5
MAX_MOVE_DURATION_S = 10.0

UNITREE_API_ERROR_CODES = {
    0: "OK",
    3001: "unknown error",
    3102: "request sending error",
    3103: "API not registered",
    3104: "request timeout",
}


def effective_loco_service_name(loco_service_name: str | None) -> str:
    value = (loco_service_name or "auto").strip()
    if value not in LOCO_SERVICE_CHOICES:
        choices = ", ".join(LOCO_SERVICE_CHOICES)
        raise UnitreeG1Error(f"Unsupported locomotion service {value!r}. Use one of: {choices}")
    return DEFAULT_G1_LOCO_SERVICE_NAME if value == "auto" else value


def loco_rpc_request_topic(loco_service_name: str) -> str:
    service = effective_loco_service_name(loco_service_name)
    return f"/api/{service}/request"


def loco_rpc_response_topic(loco_service_name: str) -> str:
    service = effective_loco_service_name(loco_service_name)
    return f"/api/{service}/response"


def unitree_error_name(code: int | None) -> str:
    if code is None:
        return "unknown"
    return UNITREE_API_ERROR_CODES.get(int(code), "unrecognized robot API error")


def unitree_rpc_result_report(result: Ros2RpcResult) -> dict[str, object]:
    return {
        "ok": result.status_code == 0,
        "api_id": result.api_id,
        "request_id": result.request_id,
        "status_code": result.status_code,
        "status_name": unitree_error_name(result.status_code),
        "data": _decode_response_data(result.data),
    }


def parse_velocity(value: str) -> tuple[float, float, float, float | None]:
    parts = value.replace(",", " ").split()
    if len(parts) not in (3, 4):
        raise ValueError("Expected velocity as 'vx vy omega' or 'vx vy omega duration'.")
    try:
        vx, vy, omega = (float(part) for part in parts[:3])
        duration = float(parts[3]) if len(parts) == 4 else None
    except ValueError as exc:
        raise ValueError("Velocity values must be numeric.") from exc
    return vx, vy, omega, duration


def _decode_response_data(raw: str) -> object:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _parameter(payload: Mapping[str, object] | None = None) -> str:
    return json.dumps(dict(payload or {}), separators=(",", ":"), allow_nan=False)


def _validate_velocity(vx: float, vy: float, omega: float, duration: float) -> None:
    values = (vx, vy, omega, duration)
    if not all(math.isfinite(value) for value in values):
        raise UnitreeG1Error("Velocity and duration values must be finite.")
    if abs(vx) > MAX_FORWARD_SPEED_MPS:
        raise UnitreeG1Error(f"vx must be within +/-{MAX_FORWARD_SPEED_MPS} m/s.")
    if abs(vy) > MAX_LATERAL_SPEED_MPS:
        raise UnitreeG1Error(f"vy must be within +/-{MAX_LATERAL_SPEED_MPS} m/s.")
    if abs(omega) > MAX_YAW_SPEED_RAD_S:
        raise UnitreeG1Error(f"omega must be within +/-{MAX_YAW_SPEED_RAD_S} rad/s.")
    if duration <= 0.0 or duration > MAX_MOVE_DURATION_S:
        raise UnitreeG1Error(f"duration must be > 0 and <= {MAX_MOVE_DURATION_S} seconds.")


RpcFactory = Callable[..., Ros2RequestResponseClient]


class G1Ros2LocoClient:
    """Bounded G1 locomotion commands over Unitree ROS 2 API topics."""

    def __init__(
        self,
        *,
        timeout_s: float = 1.0,
        loco_service_name: str = "auto",
        runner: Ros2NodeRunner | None = None,
        clients: Mapping[str, Any] | None = None,
        rpc_factory: RpcFactory = Ros2RequestResponseClient,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if loco_service_name not in LOCO_SERVICE_CHOICES:
            choices = ", ".join(LOCO_SERVICE_CHOICES)
            raise UnitreeG1Error(
                f"Unsupported locomotion service {loco_service_name!r}. Use one of: {choices}"
            )
        self.timeout_s = timeout_s
        self.loco_service_name = loco_service_name
        self._runner = runner
        self._rpc_factory = rpc_factory
        self._sleep = sleep
        self._monotonic = monotonic
        self._clients = dict(clients or {})
        self._owns_clients = clients is None
        self._owns_runner = runner is None
        self._selected_service: str | None = None
        self._started = clients is not None

    def start(self) -> "G1Ros2LocoClient":
        if self._started:
            return self
        runner = self._runner or Ros2NodeRunner("g1_locomotion_client")
        self._runner = runner
        try:
            node = runner.start()
            bindings = runner.bindings
            services = (
                ("sport", "ai_sport")
                if self.loco_service_name == "auto"
                else (self.loco_service_name,)
            )
            for service in services:
                self._clients[service] = self._rpc_factory(
                    node,
                    request_type=bindings.request_type,
                    response_type=bindings.response_type,
                    request_topic=f"/api/{service}/request",
                    response_topic=f"/api/{service}/response",
                    qos_depth=1,
                )
        except (Ros2Unavailable, ImportError, RuntimeError) as exc:
            if self._owns_runner:
                runner.close()
            raise UnitreeG1Error(f"Could not start the ROS 2 locomotion client: {exc}") from exc
        self._started = True
        return self

    def close(self) -> None:
        if self._owns_clients:
            for client in self._clients.values():
                client.close()
        self._clients.clear()
        if self._owns_runner and self._runner is not None:
            self._runner.close()
        self._started = False
        self._selected_service = None

    def __enter__(self) -> "G1Ros2LocoClient":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def probe_loco(self) -> UnitreeCommandResult:
        self.start()
        reports: dict[str, object] = {}
        selected: str | None = None
        for service in self._service_candidates():
            try:
                result = self._clients[service].call(
                    api_id=LOCO_API_GET_FSM_ID,
                    parameter=_parameter(),
                    timeout_s=self.timeout_s,
                )
                report = unitree_rpc_result_report(result)
                reports[service] = report
                if selected is None and result.status_code == 0:
                    selected = service
            except (Ros2RpcTimeout, RuntimeError) as exc:
                reports[service] = {"ok": False, "error": str(exc)}
        if selected is not None:
            self._selected_service = selected
        report = {
            "ok": selected is not None,
            "selected_service": selected,
            "read_only_calls": reports,
        }
        return self._command_result(["loco", "probe"], report, ok=selected is not None)

    def get_fsm_id(self) -> UnitreeCommandResult:
        return self._call("get_fsm_id", LOCO_API_GET_FSM_ID)

    def start_robot(self) -> UnitreeCommandResult:
        return self._call("start", LOCO_API_SET_FSM_ID, {"data": 500})

    def stand_up(self) -> UnitreeCommandResult:
        return self._call("stand_up", LOCO_API_SET_FSM_ID, {"data": 706})

    def balance_stand(self) -> UnitreeCommandResult:
        return self._call("balance_stand", LOCO_API_SET_BALANCE_MODE, {"data": 0})

    def damp(self) -> UnitreeCommandResult:
        return self._call("damp", LOCO_API_SET_FSM_ID, {"data": 1})

    def stop_move(self) -> UnitreeCommandResult:
        return self._send_velocity(0.0, 0.0, 0.0, 0.2, command_name="stop_move")

    def move(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float | None = None,
    ) -> UnitreeCommandResult:
        bounded_duration = 1.0 if duration is None else duration
        _validate_velocity(vx, vy, omega, bounded_duration)
        result = self._send_velocity(vx, vy, omega, bounded_duration, command_name="move")
        try:
            self._sleep(bounded_duration)
        finally:
            self.stop_move()
        return result

    def smooth_move(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float = 0.5,
        ramp_s: float = 0.25,
    ) -> UnitreeCommandResult:
        _validate_velocity(vx, vy, omega, duration)
        if not math.isfinite(ramp_s) or ramp_s <= 0.0:
            raise UnitreeG1Error("ramp_s must be finite and positive.")
        ramp_s = min(ramp_s, duration / 2.0)
        control_dt = 0.1
        started_at = self._monotonic()
        updates = 0
        try:
            while True:
                elapsed = self._monotonic() - started_at
                if elapsed >= duration:
                    break
                if elapsed < ramp_s:
                    scale = smoothstep(elapsed / ramp_s)
                elif elapsed > duration - ramp_s:
                    scale = smoothstep((duration - elapsed) / ramp_s)
                else:
                    scale = 1.0
                command_duration = min(0.25, max(control_dt, duration - elapsed))
                self._send_velocity(
                    vx * scale,
                    vy * scale,
                    omega * scale,
                    command_duration,
                    command_name="smooth_move_step",
                )
                updates += 1
                self._sleep(min(control_dt, max(0.0, duration - elapsed)))
        finally:
            self.stop_move()

        return self._command_result(
            ["loco", "smooth_move"],
            {
                "ok": True,
                "vx": vx,
                "vy": vy,
                "omega": omega,
                "duration": duration,
                "ramp_s": ramp_s,
                "updates": updates,
                "easing": "smoothstep",
            },
        )

    def command(self, name: str) -> UnitreeCommandResult:
        commands = {
            "start": self.start_robot,
            "stand_up": self.stand_up,
            "balance_stand": self.balance_stand,
            "stop_move": self.stop_move,
            "damp": self.damp,
            "probe_loco": self.probe_loco,
            "probe": self.probe_loco,
        }
        try:
            operation = commands[name]
        except KeyError as exc:
            choices = ", ".join(sorted(commands))
            raise UnitreeG1Error(f"Unsupported G1 command {name!r}. Use one of: {choices}") from exc
        return operation()

    def _send_velocity(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float,
        *,
        command_name: str,
    ) -> UnitreeCommandResult:
        _validate_velocity(vx, vy, omega, duration)
        return self._call(
            command_name,
            LOCO_API_SET_VELOCITY,
            {"velocity": [vx, vy, omega], "duration": duration},
        )

    def _call(
        self,
        command_name: str,
        api_id: int,
        payload: Mapping[str, object] | None = None,
    ) -> UnitreeCommandResult:
        service = self._ensure_service()
        try:
            result = self._clients[service].call(
                api_id=api_id,
                parameter=_parameter(payload),
                timeout_s=self.timeout_s,
            )
        except (Ros2RpcTimeout, RuntimeError) as exc:
            raise UnitreeG1Error(
                f"ROS 2 locomotion command {command_name!r} failed on {service}: {exc}"
            ) from exc
        report = {"service": service, **unitree_rpc_result_report(result)}
        if result.status_code != 0:
            raise UnitreeG1Error(
                f"Robot rejected {command_name!r} with status {result.status_code} "
                f"({unitree_error_name(result.status_code)})."
            )
        return self._command_result(["loco", command_name], report)

    def _ensure_service(self) -> str:
        if self._selected_service is not None:
            return self._selected_service
        probe = self.probe_loco()
        if probe.returncode != 0 or self._selected_service is None:
            raise UnitreeG1Error(
                "No responsive G1 locomotion service was found. Verify ROS 2 discovery and "
                "that the robot is in a high-level motion mode."
            )
        return self._selected_service

    def _service_candidates(self) -> tuple[str, ...]:
        if self.loco_service_name == "auto":
            return ("sport", "ai_sport")
        return (self.loco_service_name,)

    @staticmethod
    def _command_result(
        command: list[str], report: Mapping[str, object], *, ok: bool = True
    ) -> UnitreeCommandResult:
        return UnitreeCommandResult(
            command=command,
            returncode=0 if ok else 1,
            stdout=json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
            stderr="",
        )


class MotionSwitcherRos2Client:
    """ROS 2 client for read-only mode checks and explicit mode changes."""

    def __init__(
        self,
        *,
        timeout_s: float = 1.0,
        runner: Ros2NodeRunner | None = None,
        rpc_client: Any | None = None,
        rpc_factory: RpcFactory = Ros2RequestResponseClient,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._runner = runner
        self._rpc_client = rpc_client
        self._rpc_factory = rpc_factory
        self._owns_runner = runner is None
        self._owns_client = rpc_client is None

    def start(self) -> "MotionSwitcherRos2Client":
        if self._rpc_client is not None:
            return self
        runner = self._runner or Ros2NodeRunner("g1_motion_mode_client")
        self._runner = runner
        try:
            node = runner.start()
            bindings = runner.bindings
            self._rpc_client = self._rpc_factory(
                node,
                request_type=bindings.request_type,
                response_type=bindings.response_type,
                request_topic="/api/motion_switcher/request",
                response_topic="/api/motion_switcher/response",
                qos_depth=1,
            )
        except (Ros2Unavailable, ImportError, RuntimeError) as exc:
            if self._owns_runner:
                runner.close()
            raise UnitreeG1Error(f"Could not start the ROS 2 motion-mode client: {exc}") from exc
        return self

    def close(self) -> None:
        if self._owns_client and self._rpc_client is not None:
            self._rpc_client.close()
        self._rpc_client = None
        if self._owns_runner and self._runner is not None:
            self._runner.close()

    def __enter__(self) -> "MotionSwitcherRos2Client":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def check_mode(self) -> UnitreeCommandResult:
        return self._call("check_motion_mode", MOTION_API_CHECK_MODE)

    def select_mode(self, mode: str) -> UnitreeCommandResult:
        if not mode.strip():
            raise UnitreeG1Error("Motion mode is required.")
        return self._call("select_motion_mode", MOTION_API_SELECT_MODE, {"name": mode})

    def release_mode(self) -> UnitreeCommandResult:
        return self._call("release_motion_mode", MOTION_API_RELEASE_MODE)

    def _call(
        self,
        command_name: str,
        api_id: int,
        payload: Mapping[str, object] | None = None,
    ) -> UnitreeCommandResult:
        self.start()
        assert self._rpc_client is not None
        try:
            result = self._rpc_client.call(
                api_id=api_id,
                parameter=_parameter(payload),
                timeout_s=self.timeout_s,
            )
        except (Ros2RpcTimeout, RuntimeError) as exc:
            raise UnitreeG1Error(f"ROS 2 motion-mode command {command_name!r} failed: {exc}") from exc
        report = unitree_rpc_result_report(result)
        if result.status_code != 0:
            raise UnitreeG1Error(
                f"Robot rejected {command_name!r} with status {result.status_code} "
                f"({unitree_error_name(result.status_code)})."
            )
        return UnitreeCommandResult(
            command=["loco", command_name],
            returncode=0,
            stdout=json.dumps(report, indent=2, sort_keys=True) + "\n",
            stderr="",
        )
