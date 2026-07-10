"""Lazy SDK2 adapter for the persistent G1 29-DOF arm bridge."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from .arm_bridge import ArmCommand, RobotState
from .joints import ARM_INDICES, ARM_WEIGHT_INDEX


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
    ) -> None:
        self.interface = interface
        self.domain_id = domain_id
        self._monotonic = monotonic
        self.max_tilt_rad = max_tilt_rad
        self.max_standing_velocity_rad_s = max_standing_velocity_rad_s
        self._lock = threading.Lock()
        self._state: RobotState | None = None
        self._standing_since: float | None = None
        self._publisher = None
        self._subscriber = None
        self._low_cmd = None
        self._crc = None
        self._started = False

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
        except Exception as exc:  # pragma: no cover - requires robot image
            raise RuntimeError(
                "Unitree SDK2 is required on the robot for the arm bridge; install the loco dependency group"
            ) from exc

        ChannelFactoryInitialize(self.domain_id, self.interface)
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(self._low_state_callback, 10)
        self._publisher = publisher
        self._subscriber = subscriber
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._started = True

    def latest_state(self) -> RobotState | None:
        with self._lock:
            return self._state

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
        self._publisher.Write(low_cmd)

    def close(self) -> None:
        # SDK2 channel objects do not expose a consistent close method.  The
        # controller publishes the bounded zero-weight ramp before reaching us.
        self._started = False
        self._publisher = None
        self._subscriber = None
        self._low_cmd = None
        self._crc = None

    def _low_state_callback(self, message: object) -> None:
        now = self._monotonic()
        try:
            q = tuple(float(message.motor_state[index].q) for index in ARM_INDICES)
            waist_q = tuple(float(message.motor_state[index].q) for index in (12, 13, 14))
            body_velocity = tuple(float(message.motor_state[index].dq) for index in range(29))
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
            still = all(
                math.isfinite(value) and abs(value) <= self.max_standing_velocity_rad_s
                for value in body_velocity
            )
            finite = all(math.isfinite(value) for value in q)
            standing = bool(level and still and finite)
        except (AttributeError, IndexError, TypeError, ValueError):
            q = (math.nan,) * 14
            waist_q = (math.nan,) * 3
            mode_machine = 0
            standing = False

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
            )
