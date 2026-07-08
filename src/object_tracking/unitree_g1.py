from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
import time


try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except Exception as exc:  # pragma: no cover - surfaced as runtime error when dependency missing
    ChannelFactoryInitialize = None
    LocoClient = None
    _SDK2_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SDK2_IMPORT_ERROR = None

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except Exception as exc:  # pragma: no cover - only needed for low-level arm test
    ChannelPublisher = None
    ChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    LowCmd_ = None
    LowState_ = None
    CRC = None
    _SDK2_LOW_LEVEL_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SDK2_LOW_LEVEL_IMPORT_ERROR = None


def default_camera_env() -> list[str]:
    value = os.environ.get("G1_CAMERA_SOURCES")
    if value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def default_camera_ports() -> list[str]:
    value = os.environ.get("G1_CAMERA_UDP_PORTS", "5600")
    ports = []
    for part in value.split(","):
        p = part.strip()
        if p:
            ports.append(p)
    if not ports:
        ports.append("5600")
    return ports


def default_gstreamer_port_templates() -> list[str]:
    return [
        "udpsrc address={ip} port={port} ! application/x-rtp,media=(string)video,encoding-name=(string)H264,clock-rate=(int)90000,payload=(int)96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink",
        "udpsrc address={ip} port={port} ! application/x-rtp,media=(string)video ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink",
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


def normalize_network_interface(network_interface: str | None) -> str | None:
    if network_interface is None:
        return None
    value = network_interface.strip()
    if not value or value.lower() == "auto":
        return None
    return value


class UnitreeSdk2Context:
    _initialized = False
    _domain_id: int | None = None
    _network_interface: str | None = None

    @classmethod
    def initialize(
        cls,
        robot_ip: str | None = None,
        network_interface: str | None = None,
        domain_id: int = 0,
    ) -> str:
        if ChannelFactoryInitialize is None:
            detail = str(_SDK2_IMPORT_ERROR) if _SDK2_IMPORT_ERROR is not None else "missing imports"
            raise UnitreeG1Error(
                "Could not import unitree-sdk2 Python package. Install dependencies with: `uv sync`.\n"
                f"Original error: {detail}"
            )

        resolved_interface = normalize_network_interface(network_interface)
        if resolved_interface is None:
            if robot_ip is None:
                raise UnitreeG1Error("Pass robot_ip or --interface/--network-interface for G1 SDK2 commands.")
            resolved_interface = resolve_network_interface(robot_ip)

        if cls._initialized:
            if cls._domain_id != domain_id or cls._network_interface != resolved_interface:
                raise UnitreeG1Error(
                    "Unitree SDK2 DDS is already initialized in this Python process with "
                    f"domain={cls._domain_id}, interface={cls._network_interface!r}; "
                    f"cannot reinitialize to domain={domain_id}, interface={resolved_interface!r}. "
                    "Run a fresh process for a different DDS interface/domain."
                )
            return resolved_interface

        try:
            ChannelFactoryInitialize(domain_id, resolved_interface)
        except Exception as exc:
            raise UnitreeG1Error(
                "Failed to initialize Unitree SDK2 DDS.\n"
                f"Selected interface: {resolved_interface}\n"
                f"Selected domain: {domain_id}\n"
                "Try:\n"
                f"  ip route get {robot_ip or '<robot_ip>'}\n"
                f"  uv run loco {robot_ip or '<robot_ip>'} stop_move --interface <dev>\n"
                f"  uv run loco --diagnose {robot_ip or '<robot_ip>'}"
            ) from exc

        cls._initialized = True
        cls._domain_id = domain_id
        cls._network_interface = resolved_interface
        return resolved_interface


def g1_loco_init_error(exc: Exception, robot_ip: str | None, network_interface: str, domain_id: int) -> UnitreeG1Error:
    robot = robot_ip or "<robot_ip>"
    return UnitreeG1Error(
        "Failed to initialize Unitree G1 LocoClient before sending command.\n\n"
        "Likely causes:\n"
        "1. CycloneDDS topic/type conflict on Unitree RPC topic rt/api/sport/request.\n"
        "2. Wrong local network interface selected for CycloneDDS.\n"
        "3. unitree_sdk2py checkout/version mismatch with this repo or the robot firmware.\n\n"
        "Try:\n"
        f"  ip route get {robot}\n"
        f"  uv run loco --diagnose {robot}\n"
        f"  uv run loco {robot} stop_move --interface {network_interface}\n\n"
        f"DDS domain: {domain_id}\n"
        f"DDS interface: {network_interface}\n"
        f"Original error: {exc}"
    )


def g1_camera_candidates(robot_ip: str, camera_url: str | None = None) -> list[str]:
    explicit_url = camera_url or os.environ.get("G1_CAMERA_URL")
    if explicit_url:
        return [explicit_url.format(ip=robot_ip)]

    configured = default_camera_env()
    if configured:
        return [candidate.format(ip=robot_ip) for candidate in configured]

    return [template.format(ip=robot_ip) for template in DEFAULT_G1_CAMERA_TEMPLATES]


class G1LocoSdk2Client:
    """Thin wrapper around Unitree SDK2 Python G1 loco client."""

    def __init__(
        self,
        network_interface: str | None = None,
        robot_ip: str | None = None,
        timeout_s: float = 15.0,
        domain_id: int = 0,
    ) -> None:
        if LocoClient is None:
            detail = str(_SDK2_IMPORT_ERROR) if _SDK2_IMPORT_ERROR is not None else "missing imports"
            raise UnitreeG1Error(
                "Could not import unitree-sdk2 Python package. Install dependencies with: `uv sync`.\n"
                f"Original error: {detail}"
            )

        self.network_interface = UnitreeSdk2Context.initialize(
            robot_ip=robot_ip,
            network_interface=network_interface,
            domain_id=domain_id,
        )
        self.timeout_s = timeout_s
        self.domain_id = domain_id

        try:
            self._client = LocoClient()
        except Exception as exc:
            raise g1_loco_init_error(exc, robot_ip, self.network_interface, domain_id) from exc
        self._client.Init()
        self._client.SetTimeout(timeout_s)
        self._arm_low_state = None

    def _result(self, command: list[str], code: int) -> UnitreeCommandResult:
        code_int = int(code)
        if code_int != 0:
            raise UnitreeG1Error(f"unitree-sdk2 command returned non-zero code {code_int}: {command}")
        return UnitreeCommandResult(command=command, returncode=0, stdout="", stderr="")

    def _call(self, method: str, *args: object) -> UnitreeCommandResult:
        method_obj = getattr(self._client, method)
        try:
            call_result = method_obj(*args) if args else method_obj()
        except TypeError:
            if method == "BalanceStand" and not args:
                call_result = method_obj(0)
            else:
                raise
        if isinstance(call_result, tuple) and len(call_result) >= 1:
            code = call_result[0]
        else:
            code = call_result
        return self._result(["loco", method.lower()], code if code is not None else 0)

    def _call_first(self, methods: list[str]) -> UnitreeCommandResult:
        for method in methods:
            if hasattr(self._client, method):
                return self._call(method)
        choices = ", ".join(methods)
        raise UnitreeG1Error(f"Installed G1 loco client does not expose any of: {choices}")

    def get_fsm_id(self) -> UnitreeCommandResult:
        if hasattr(self._client, "GetFsmId"):
            result = self._client.GetFsmId()
            code = result[0] if isinstance(result, tuple) and result else result
            stdout = ""
            if isinstance(result, tuple) and len(result) > 1:
                stdout = f"{result[1]}\n"
            return UnitreeCommandResult(command=["loco", "get_fsm_id"], returncode=int(code or 0), stdout=stdout, stderr="")
        raise UnitreeG1Error("get_fsm_id is not implemented by the installed G1 loco client.")

    def start(self) -> UnitreeCommandResult:
        return self._call("Start")

    def stand_up(self) -> UnitreeCommandResult:
        return self._call_first(["StandUp", "Squat2StandUp", "Lie2StandUp", "HighStand"])

    def balance_stand(self) -> UnitreeCommandResult:
        return self._call("BalanceStand")

    def stop_move(self) -> UnitreeCommandResult:
        return self._call("StopMove")

    def damp(self) -> UnitreeCommandResult:
        return self._call("Damp")

    def move(self, vx: float, vy: float, omega: float, duration: float | None = None) -> UnitreeCommandResult:
        cmd = ["loco", "move", str(vx), str(vy), str(omega)]
        if duration is not None:
            cmd.append(str(duration))

        result = self._call("Move", vx, vy, omega)

        if duration is not None:
            time.sleep(duration)
            self.stop_move()

        return result

    def _low_state_handler(self, msg: object) -> None:
        self._arm_low_state = msg

    def move_arms_up(self) -> UnitreeCommandResult:
        if (
            ChannelPublisher is None
            or ChannelSubscriber is None
            or unitree_hg_msg_dds__LowCmd_ is None
            or LowCmd_ is None
            or LowState_ is None
            or CRC is None
        ):
            detail = (
                str(_SDK2_LOW_LEVEL_IMPORT_ERROR)
                if _SDK2_LOW_LEVEL_IMPORT_ERROR is not None
                else "missing low-level SDK2 imports"
            )
            raise UnitreeG1Error(f"Could not import SDK2 low-level G1 DDS APIs.\nOriginal error: {detail}")

        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(self._low_state_handler, 10)

        deadline = time.monotonic() + min(self.timeout_s, 10.0)
        while self._arm_low_state is None:
            if time.monotonic() > deadline:
                raise UnitreeG1Error("Timed out waiting for rt/lowstate before moving arms.")
            time.sleep(0.01)

        low_cmd = unitree_hg_msg_dds__LowCmd_()
        crc = CRC()
        arm_joint_ids = sorted(G1_ARM_FORWARD_TARGETS)
        start_q = {
            joint_id: float(self._arm_low_state.motor_state[joint_id].q)
            for joint_id in arm_joint_ids
        }

        control_dt = 0.02
        ramp_s = 2.0
        started_at = time.monotonic()

        try:
            while True:
                elapsed = time.monotonic() - started_at
                ratio = min(max(elapsed / ramp_s, 0.0), 1.0)
                low_cmd.mode_pr = 0
                low_cmd.mode_machine = int(getattr(self._arm_low_state, "mode_machine", 0))

                for joint_id in arm_joint_ids:
                    target = G1_ARM_FORWARD_TARGETS[joint_id]
                    motor = low_cmd.motor_cmd[joint_id]
                    motor.mode = 1
                    motor.tau = 0.0
                    motor.q = (1.0 - ratio) * start_q[joint_id] + ratio * target
                    motor.dq = 0.0
                    motor.kp = 25.0
                    motor.kd = 1.0

                low_cmd.crc = crc.Crc(low_cmd)
                publisher.Write(low_cmd)
                time.sleep(control_dt)
        except KeyboardInterrupt:
            return UnitreeCommandResult(command=["loco", "move_arms_up"], returncode=0, stdout="", stderr="")

    def command(self, name: str) -> UnitreeCommandResult:
        commands = {
            "stand_up": self.stand_up,
            "balance_stand": self.balance_stand,
            "stop_move": self.stop_move,
            "damp": self.damp,
            "move_arms_up": self.move_arms_up,
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
