from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
import time


try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient
except Exception as exc:  # pragma: no cover - surfaced as runtime error when dependency missing
    ChannelFactoryInitialize = None
    SportClient = None
    VideoClient = None
    _SDK2_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SDK2_IMPORT_ERROR = None


def default_camera_env() -> list[str]:
    value = os.environ.get("G1_CAMERA_SOURCES")
    if value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def default_camera_ports() -> list[str]:
    value = os.environ.get("G1_CAMERA_UDP_PORTS", "1720")
    ports = []
    for part in value.split(","):
        p = part.strip()
        if p:
            ports.append(p)
    if not ports:
        ports.append("1720")
    return ports


def default_gstreamer_port_templates() -> list[str]:
    multicast_group = os.environ.get("G1_CAMERA_MULTICAST_GROUP", "230.1.1.1").strip()
    multicast_iface = os.environ.get("G1_CAMERA_MULTICAST_IFACE", "").strip()
    iface = f" multicast-iface={multicast_iface}" if multicast_iface else ""
    return [
        f"udpsrc multicast-group={multicast_group} address=0.0.0.0 port={{port}} auto-multicast=true{iface} ! application/x-rtp,media=(string)video,encoding-name=(string)H264,clock-rate=(int)90000 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink",
        f"udpsrc address={multicast_group} port={{port}} auto-multicast=true ! application/x-rtp,media=(string)video,encoding-name=(string)H264,clock-rate=(int)90000 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink",
    ]


def default_camera_sources() -> list[str]:
    transport = os.environ.get("G1_CAMERA_TRANSPORT", "gstreamer").lower()
    ports = default_camera_ports()
    if transport == "rtsp":
        templates = (
            "rtsp://{ip}:8554/unicast",
            "rtsp://{ip}:8554/live",
            "rtsp://{ip}:8554/stream",
            "rtsp://{ip}:554/live",
            "rtsp://{ip}:554/stream1",
            "rtsp://{ip}:554/h264Preview_01_main",
        )
        return [template for template in templates]

    result: list[str] = []
    for template in default_gstreamer_port_templates():
        for port in ports:
            result.append(template.format(port=port, ip="{ip}"))
    return result


def is_gstreamer_pipeline(source: str) -> bool:
    return isinstance(source, str) and "!" in source and ("udpsrc" in source or "gst-launch" in source)


DEFAULT_G1_CAMERA_TEMPLATES = tuple(default_camera_sources())


class UnitreeG1Error(RuntimeError):
    pass


@dataclass(frozen=True)
class UnitreeCommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def resolve_network_interface(robot_ip: str) -> str:
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnitreeG1Error(f"Could not resolve route to robot IP {robot_ip!r}") from exc

    if completed.returncode != 0:
        raise UnitreeG1Error(
            f"No route to robot IP {robot_ip!r}. Connect to the robot network first.\n"
            f"{completed.stderr.strip()}"
        )

    tokens = completed.stdout.split()
    if "dev" not in tokens:
        raise UnitreeG1Error(f"Could not find interface in route output: {completed.stdout.strip()}")
    dev_index = tokens.index("dev") + 1
    if dev_index >= len(tokens):
        raise UnitreeG1Error(f"Malformed route output: {completed.stdout.strip()}")
    return tokens[dev_index]


def g1_camera_candidates(robot_ip: str, camera_url: str | None = None) -> list[str]:
    explicit_url = camera_url or os.environ.get("G1_CAMERA_URL")
    if explicit_url:
        return [explicit_url.format(ip=robot_ip)]

    configured = default_camera_env()
    if configured:
        return [candidate.format(ip=robot_ip) for candidate in configured]

    return [template.format(ip=robot_ip) for template in DEFAULT_G1_CAMERA_TEMPLATES]


def get_sdk2_video_sample(
    robot_ip: str | None = None,
    network_interface: str | None = None,
    timeout_s: float = 3.0,
) -> tuple[bytes, str]:
    if ChannelFactoryInitialize is None or VideoClient is None:
        detail = str(_SDK2_IMPORT_ERROR) if _SDK2_IMPORT_ERROR is not None else "missing imports"
        raise UnitreeG1Error(
            "Could not import Unitree SDK2 video dependencies. Install and configure unitree-sdk2py and CycloneDDS on the robot-network host.\n"
            f"Original error: {detail}"
        )

    if network_interface is None:
        if robot_ip is None:
            raise UnitreeG1Error("Pass robot_ip or network_interface for SDK2 video.")
        network_interface = resolve_network_interface(robot_ip)

    ChannelFactoryInitialize(0, network_interface)
    client = VideoClient()
    client.Init()
    client.SetTimeout(timeout_s)

    code, image = client.GetImageSample()
    code_int = int(code) if code is not None else -1
    if code_int != 0:
        raise UnitreeG1Error(f"SDK2 videohub GetImageSample returned code {code_int}.")
    if image is None or len(image) == 0:
        raise UnitreeG1Error("SDK2 videohub GetImageSample returned no image bytes.")

    return bytes(image), network_interface


class G1LocoSdk2Client:
    """Thin wrapper around Unitree SDK2 Python sport client (unitree-sdk2py==1.0.1)."""

    def __init__(
        self,
        network_interface: str | None = None,
        robot_ip: str | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        if ChannelFactoryInitialize is None or SportClient is None:
            detail = str(_SDK2_IMPORT_ERROR) if _SDK2_IMPORT_ERROR is not None else "missing imports"
            raise UnitreeG1Error(
                "Could not import unitree-sdk2py Python package. Install loco dependencies with: `uv sync`.\n"
                f"Original error: {detail}"
            )

        if network_interface is None:
            if robot_ip is None:
                raise UnitreeG1Error("Pass robot_ip or network_interface for G1 SDK2 commands.")
            network_interface = resolve_network_interface(robot_ip)

        self.network_interface = network_interface
        self.timeout_s = timeout_s

        ChannelFactoryInitialize(0, network_interface)
        self._client = SportClient()
        self._client.Init()
        self._client.SetTimeout(timeout_s)

    def _result(self, command: list[str], code: int) -> UnitreeCommandResult:
        code_int = int(code)
        if code_int != 0:
            raise UnitreeG1Error(f"unitree-sdk2py command returned non-zero code {code_int}: {command}")
        return UnitreeCommandResult(command=command, returncode=0, stdout="", stderr="")

    def _call(self, method: str, *args: object) -> UnitreeCommandResult:
        method_obj = getattr(self._client, method)
        call_result = method_obj(*args) if args else method_obj()
        if isinstance(call_result, tuple) and len(call_result) >= 1:
            code = call_result[0]
        else:
            code = call_result
        return self._result(["sport", method.lower()], code if code is not None else 0)

    def get_fsm_id(self) -> UnitreeCommandResult:
        raise UnitreeG1Error("get_fsm_id is not implemented in unitree-sdk2py. Use stand_up/balance_stand/move/specific sport commands.")

    def start(self) -> UnitreeCommandResult:
        raise UnitreeG1Error("start is not implemented in unitree-sdk2py. Use stand_up or balance_stand.")

    def stand_up(self) -> UnitreeCommandResult:
        return self._call("StandUp")

    def balance_stand(self) -> UnitreeCommandResult:
        return self._call("BalanceStand")

    def stop_move(self) -> UnitreeCommandResult:
        return self._call("StopMove")

    def damp(self) -> UnitreeCommandResult:
        return self._call("Damp")

    def move(self, vx: float, vy: float, omega: float, duration: float | None = None) -> UnitreeCommandResult:
        cmd = ["sport", "move", str(vx), str(vy), str(omega)]
        if duration is not None:
            cmd.append(str(duration))

        result = self._call("Move", vx, vy, omega)

        if duration is not None:
            time.sleep(duration)
            self.stop_move()

        return result

    def command(self, name: str) -> UnitreeCommandResult:
        commands = {
            "stand_up": self.stand_up,
            "balance_stand": self.balance_stand,
            "stop_move": self.stop_move,
            "damp": self.damp,
        }
        try:
            return commands[name]()
        except KeyError as exc:
            choices = ", ".join(sorted(commands))
            raise UnitreeG1Error(f"Unsupported G1 command {name!r}. Use one of: {choices}") from exc


def parse_velocity(value: str) -> tuple[float, float, float, float | None]:
    parts = value.split()
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("Expected 'vx vy omega' or 'vx vy omega duration'.")
    vx, vy, omega = (float(part) for part in parts[:3])
    duration = float(parts[3]) if len(parts) == 4 else None
    return vx, vy, omega, duration
