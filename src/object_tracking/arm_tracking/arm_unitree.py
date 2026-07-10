"""Lazy SDK2 adapter for the persistent G1 29-DOF arm bridge."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
import threading
import time
from typing import Callable

from .arm_bridge import ArmCommand, RobotState
from .joints import ARM_INDICES, ARM_WEIGHT_INDEX, BODY_JOINT_INDICES


class UnitreeArmHardware:
    """One publisher/subscriber pair kept alive for the bridge lifetime.

    SDK imports and DDS initialization intentionally happen only in ``start``;
    importing the service on GB10 or in tests therefore needs no robot package.
    """

    def __init__(
        self,
        *,
        interface: str = "wlan0",
        domain_id: int = 0,
        monotonic: Callable[[], float] = time.monotonic,
        max_tilt_rad: float = math.radians(5.0),
        max_standing_velocity_rad_s: float = 0.25,
        max_angular_rate_rad_s: float = 0.5,
        expected_motion_mode: str | None = None,
        ownership_quiet_s: float = 1.0,
    ) -> None:
        self.interface = interface
        self.domain_id = domain_id
        self._monotonic = monotonic
        self.max_tilt_rad = max_tilt_rad
        self.max_standing_velocity_rad_s = max_standing_velocity_rad_s
        self.max_angular_rate_rad_s = max_angular_rate_rad_s
        self.expected_motion_mode = expected_motion_mode
        self.ownership_quiet_s = ownership_quiet_s
        self._lock = threading.Lock()
        self._state: RobotState | None = None
        self._standing_since: float | None = None
        self._publisher = None
        self._subscriber = None
        self._arm_subscriber = None
        self._low_cmd = None
        self._crc = None
        self._started = False
        self._started_at: float | None = None
        self._published_fingerprints: deque[tuple[float, ...]] = deque(maxlen=32)
        self._ownership_conflict = False
        self._motion_mode_name: str | None = None
        self._motion_mode_checked = False
        self._motion_client = None
        self._motion_thread: threading.Thread | None = None
        self._motion_stop = threading.Event()

    def start(self) -> None:
        if self._started:
            return
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
        except Exception as exc:  # pragma: no cover - requires robot image
            raise RuntimeError(
                "Unitree SDK2 is required on the robot for the arm bridge; install the loco dependency group"
            ) from exc

        ChannelFactoryInitialize(self.domain_id, self.interface)
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(self._low_state_callback, 10)
        arm_subscriber = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        arm_subscriber.Init(self._arm_command_callback, 10)
        motion_client = MotionSwitcherClient()
        motion_client.SetTimeout(1.0)
        motion_client.Init()
        self._publisher = publisher
        self._subscriber = subscriber
        self._arm_subscriber = arm_subscriber
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._motion_client = motion_client
        self._started_at = self._monotonic()
        self._motion_stop.clear()
        self._motion_thread = threading.Thread(
            target=self._poll_motion_mode,
            daemon=True,
            name="unitree-motion-mode-monitor",
        )
        self._motion_thread.start()
        self._started = True

    def latest_state(self) -> RobotState | None:
        with self._lock:
            state = self._state
            if state is None:
                return None
            now = self._monotonic()
            ownership_verified = bool(
                self._started_at is not None
                and now - self._started_at >= self.ownership_quiet_s
                and not self._ownership_conflict
            )
            mode_verified = bool(
                self.expected_motion_mode is not None
                and self._motion_mode_checked
                and self._motion_mode_name == self.expected_motion_mode
            )
            return replace(
                state,
                compatible_motion_mode=mode_verified,
                controller_available=ownership_verified,
                motion_mode_name=self._motion_mode_name,
                motion_mode_verified=mode_verified,
                controller_ownership_verified=ownership_verified,
            )

    def publish(self, command: ArmCommand) -> None:
        if (
            not self._started
            or self._publisher is None
            or self._low_cmd is None
            or self._crc is None
        ):
            raise RuntimeError("Unitree arm hardware has not been started")
        low_cmd = self._low_cmd
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = command.mode_machine
        for offset, joint_index in enumerate(ARM_INDICES):
            motor = low_cmd.motor_cmd[joint_index]
            motor.mode = 1
            motor.q = command.q[offset]
            motor.dq = command.dq[offset]
            motor.kp = command.kp[offset]
            motor.kd = command.kd[offset]
            motor.tau = 0.0
        # Unitree's arm_sdk convention uses the not-used motor slot as an arm
        # arbitration weight.  Waist and leg command slots remain untouched.
        low_cmd.motor_cmd[ARM_WEIGHT_INDEX].q = command.weight
        low_cmd.crc = self._crc.Crc(low_cmd)
        fingerprint = tuple(command.q) + (float(command.weight), float(command.mode_machine))
        with self._lock:
            self._published_fingerprints.append(fingerprint)
        self._publisher.Write(low_cmd)

    def close(self) -> None:
        # SDK2 channel objects do not expose a consistent close method.  The
        # controller publishes the bounded zero-weight ramp before reaching us.
        self._started = False
        self._motion_stop.set()
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=1.5)
        self._publisher = None
        self._subscriber = None
        self._arm_subscriber = None
        self._low_cmd = None
        self._crc = None
        self._motion_client = None
        self._motion_thread = None

    def _poll_motion_mode(self) -> None:
        while not self._motion_stop.is_set():
            client = self._motion_client
            try:
                result = client.CheckMode() if client is not None else None
                name: str | None = None
                if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
                    name = str(result[1].get("name") or "")
                with self._lock:
                    self._motion_mode_name = name
                    self._motion_mode_checked = name is not None
            except Exception:
                with self._lock:
                    self._motion_mode_checked = False
            self._motion_stop.wait(0.1)

    def _arm_command_callback(self, message: object) -> None:
        try:
            fingerprint = tuple(float(message.motor_cmd[index].q) for index in ARM_INDICES) + (
                float(message.motor_cmd[ARM_WEIGHT_INDEX].q),
                float(getattr(message, "mode_machine", 0)),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            with self._lock:
                self._ownership_conflict = True
            return
        with self._lock:
            matched_ours = any(
                all(
                    abs(actual - wanted) <= 1e-4
                    for actual, wanted in zip(fingerprint, expected, strict=True)
                )
                for expected in self._published_fingerprints
            )
            if not matched_ours:
                self._ownership_conflict = True

    def _low_state_callback(self, message: object) -> None:
        now = self._monotonic()
        try:
            body_q = tuple(float(message.motor_state[index].q) for index in BODY_JOINT_INDICES)
            body_dq = tuple(float(message.motor_state[index].dq) for index in BODY_JOINT_INDICES)
            q = tuple(float(message.motor_state[index].q) for index in ARM_INDICES)
            arm_dq = tuple(float(message.motor_state[index].dq) for index in ARM_INDICES)
            waist_q = tuple(float(message.motor_state[index].q) for index in (12, 13, 14))
            balance_velocity = tuple(float(message.motor_state[index].dq) for index in range(15))
            mode_machine = int(getattr(message, "mode_machine", 0))
            imu = getattr(message, "imu_state", None)
            rpy = tuple(
                float(value) for value in getattr(imu, "rpy", (math.inf, math.inf, math.inf))
            )
            level = (
                len(rpy) >= 2
                and abs(rpy[0]) <= self.max_tilt_rad
                and abs(rpy[1]) <= self.max_tilt_rad
            )
            angular_rate = tuple(
                float(value)
                for value in getattr(
                    imu,
                    "gyroscope",
                    getattr(imu, "omega", (math.inf, math.inf, math.inf)),
                )
            )
            angularly_still = len(angular_rate) >= 3 and all(
                math.isfinite(value) and abs(value) <= self.max_angular_rate_rad_s
                for value in angular_rate[:3]
            )
            still = all(
                math.isfinite(value) and abs(value) <= self.max_standing_velocity_rad_s
                for value in balance_velocity
            )
            finite = all(math.isfinite(value) for value in (*body_q, *body_dq))
            motor_faults: list[str] = []
            motor_status_verified = False
            for index in range(29):
                motor = message.motor_state[index]
                temperature = getattr(motor, "temperature", None)
                if isinstance(temperature, (int, float)):
                    motor_status_verified = True
                    if not math.isfinite(float(temperature)) or float(temperature) > 85.0:
                        motor_faults.append(f"motor_{index}_temperature")
                lost = getattr(motor, "lost", None)
                if isinstance(lost, (int, bool)):
                    motor_status_verified = True
                    if int(lost) != 0:
                        motor_faults.append(f"motor_{index}_lost")
            motor_state_healthy = motor_status_verified and not motor_faults
            standing = bool(level and angularly_still and still and finite)
            balance_details = tuple(
                name
                for name, passed in (
                    ("imu_level", level),
                    ("imu_angular_rate", angularly_still),
                    ("legs_waist_still", still),
                    ("arm_state_finite", finite),
                )
                if not passed
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            body_q = (math.nan,) * 29
            body_dq = (math.nan,) * 29
            q = (math.nan,) * 14
            arm_dq = (math.nan,) * 14
            waist_q = (math.nan,) * 3
            mode_machine = 0
            standing = False
            balance_details = ("lowstate_parse_error",)
            motor_status_verified = False
            motor_state_healthy = False
            motor_faults = ["lowstate_parse_error"]

        with self._lock:
            if standing:
                if self._standing_since is None:
                    self._standing_since = now
            else:
                self._standing_since = None
            self._state = RobotState(
                arm_q=q,
                received_at=now,
                standing=standing,
                standing_since=self._standing_since,
                # The bridge owns a single process-level publisher; a future
                # firmware arbitration signal can tighten these two flags.
                compatible_motion_mode=True,
                controller_available=True,
                mode_machine=mode_machine,
                waist_q=waist_q,
                arm_dq=arm_dq,
                motor_status_verified=motor_status_verified,
                motor_state_healthy=motor_state_healthy,
                motor_faults=tuple(motor_faults),
                balance_details=balance_details,
                body_q=body_q,
                body_dq=body_dq,
            )
