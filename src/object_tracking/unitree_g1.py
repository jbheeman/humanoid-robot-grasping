from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SDK2_PATH = Path("~/Documents/unitree_sdk2").expanduser()
DEFAULT_G1_CAMERA_TEMPLATES = (
    "rtsp://{ip}:8554/unicast",
    "rtsp://{ip}:8554/live",
    "rtsp://{ip}:8554/stream",
    "rtsp://{ip}:554/live",
    "rtsp://{ip}:554/stream1",
    "rtsp://{ip}:554/h264Preview_01_main",
)


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
    return [template.format(ip=robot_ip) for template in DEFAULT_G1_CAMERA_TEMPLATES]


class G1LocoSdk2Client:
    """Thin Python wrapper around the SDK2 G1 loco example executable."""

    def __init__(
        self,
        network_interface: str | None = None,
        robot_ip: str | None = None,
        sdk2_path: Path = DEFAULT_SDK2_PATH,
        loco_binary: Path | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        if network_interface is None:
            if robot_ip is None:
                raise UnitreeG1Error("Pass robot_ip or network_interface for G1 SDK2 commands.")
            network_interface = resolve_network_interface(robot_ip)

        self.robot_ip = robot_ip
        self.network_interface = network_interface
        self.sdk2_path = sdk2_path.expanduser()
        self.timeout_s = timeout_s
        self.loco_binary = (
            loco_binary.expanduser() if loco_binary is not None else self._find_loco_binary()
        )

    def get_fsm_id(self) -> UnitreeCommandResult:
        return self._run("--get_fsm_id")

    def start(self) -> UnitreeCommandResult:
        return self._run("--start")

    def stand_up(self) -> UnitreeCommandResult:
        return self._run("--stand_up")

    def balance_stand(self) -> UnitreeCommandResult:
        return self._run("--balance_stand")

    def stop_move(self) -> UnitreeCommandResult:
        return self._run("--stop_move")

    def damp(self) -> UnitreeCommandResult:
        return self._run("--damp")

    def move(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float | None = None,
    ) -> UnitreeCommandResult:
        values = [vx, vy, omega]
        if duration is not None:
            values.append(duration)
        payload = " ".join(str(value) for value in values)
        return self._run(f"--set_velocity={payload}")

    def command(self, name: str) -> UnitreeCommandResult:
        commands = {
            "get_fsm_id": self.get_fsm_id,
            "start": self.start,
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

    def _find_loco_binary(self) -> Path:
        candidates = (
            self.sdk2_path / "build" / "bin" / "g1_loco_client",
            self.sdk2_path / "build" / "g1_loco_client",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise UnitreeG1Error(
            "Could not find SDK2 g1_loco_client. Build it with: "
            f"cd {self.sdk2_path} && mkdir -p build && cd build && cmake .. && make g1_loco_client"
        )

    def _run(self, *args: str) -> UnitreeCommandResult:
        command = [
            str(self.loco_binary),
            f"--network_interface={self.network_interface}",
            *args,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            rendered = shlex.join(command)
            raise UnitreeG1Error(f"Timed out running G1 SDK2 command: {rendered}") from exc

        result = UnitreeCommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if completed.returncode != 0:
            rendered = shlex.join(command)
            raise UnitreeG1Error(
                f"G1 SDK2 command failed with exit code {completed.returncode}: {rendered}\n"
                f"{completed.stderr.strip()}"
            )
        return result


def parse_velocity(value: str) -> tuple[float, float, float, float | None]:
    parts = value.split()
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("Expected 'vx vy omega' or 'vx vy omega duration'.")
    vx, vy, omega = (float(part) for part in parts[:3])
    duration = float(parts[3]) if len(parts) == 4 else None
    return vx, vy, omega, duration
