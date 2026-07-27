"""Robot-local Unitree SDK adapter for the persistent G1 29-DOF arm bridge.

The G1 firmware publishes ``rt/lowstate`` and consumes ``rt/arm_sdk`` using
Unitree's native DDS IDL.  It is *not* a ROS 2 ``unitree_hg`` endpoint: trying
to subscribe with generated ROS messages discovers the topic but cannot decode
samples on the stock robot image.  Project traffic stays on ROS 2; only this
robot-local hardware edge uses SDK2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import threading
import time
from typing import Callable, Optional

from .arm_bridge import ArmCommand, RobotState
from .joints import ARM_INDICES, ARM_WEIGHT_INDEX, BODY_JOINT_INDICES


LOW_STATE_TOPIC = "rt/lowstate"
ARM_COMMAND_TOPIC = "rt/arm_sdk"
# Kept as deprecated import aliases for callers that previously imported the
# ROS motion-switcher topics from this module.  Native SDK2 owns motion-mode
# checks now, so these are no longer used by the hardware implementation.
MOTION_REQUEST_TOPIC = "/api/motion_switcher/request"
MOTION_RESPONSE_TOPIC = "/api/motion_switcher/response"

# Gains used by Unitree's current motion-mode G1 arm controller reference.
_WRIST_INDICES = frozenset((19, 20, 21, 26, 27, 28))
_WEAK_INDICES = frozenset((4, 10, 15, 16, 17, 18, 22, 23, 24, 25))


class UnitreeArmHardware:
    """Native-SDK implementation of the bridge's small ``ArmHardware`` API."""

    def __init__(
        self,
        *,
        interface: str = "eth0",
        domain_id: int = 0,
        monotonic: Callable[[], float] = time.monotonic,
        max_tilt_rad: float = math.radians(5.0),
        max_standing_velocity_rad_s: float = 0.25,
        max_angular_rate_rad_s: float = 0.5,
        expected_motion_mode: Optional[str] = None,
        ownership_quiet_s: float = 1.0,
        motion_poll_s: float = 0.25,
        motion_mode_grace_s: float = 30.0,
    ) -> None:
        if ownership_quiet_s < 0.0:
            raise ValueError("ownership_quiet_s must be non-negative")
        if motion_poll_s <= 0.0:
            raise ValueError("motion_poll_s must be positive")
        if motion_mode_grace_s <= 0.0:
            raise ValueError("motion_mode_grace_s must be positive")
        self.interface = interface
        self.domain_id = domain_id
        self._monotonic = monotonic
        self.max_tilt_rad = max_tilt_rad
        self.max_standing_velocity_rad_s = max_standing_velocity_rad_s
        self.max_angular_rate_rad_s = max_angular_rate_rad_s
        self.expected_motion_mode = expected_motion_mode
        self.ownership_quiet_s = ownership_quiet_s
        self.motion_poll_s = motion_poll_s
        self.motion_mode_grace_s = motion_mode_grace_s
        self._lock = threading.Lock()
        self._state: Optional[RobotState] = None
        self._standing_since: Optional[float] = None
        self._publisher: Optional[object] = None
        self._subscriber: Optional[object] = None
        self._arm_subscriber: Optional[object] = None
        self._low_cmd: Optional[object] = None
        self._motion_client: Optional[object] = None
        self._crc: Optional[object] = None
        self._started = False
        self._started_at: Optional[float] = None
        self._published_fingerprints: deque[tuple[float, ...]] = deque(maxlen=32)
        self._ownership_conflict = False
        self._motion_mode_name: Optional[str] = None
        self._motion_mode_checked = False
        self._motion_mode_verified_at: Optional[float] = None
        self._motion_mode_error: Optional[str] = None
        self._motion_thread: Optional[threading.Thread] = None
        self._motion_stop = threading.Event()

    def start(self) -> None:
        if self._started:
            return
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
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
                "Native Unitree SDK2 import failed: "
                f"{exc}. Check UNITREE_SDK_PYTHONPATH and "
                "UNITREE_SDK_DDS_LIBRARY_DIR."
            ) from exc

        # SDK2 accepts one NIC.  ROS may use multiple NICs for GB10 transport;
        # native Unitree motor DDS is intentionally kept on the robot NIC.
        interface = self.interface.split(",", 1)[0].strip()
        if not interface:
            raise ValueError("native Unitree interface must not be empty")
        # Match xr_teleoperate's real-robot initialization exactly; the
        # keyword form selects the intended native DDS NIC on the stock G1.
        ChannelFactoryInitialize(self.domain_id, networkInterface=interface)
        publisher = ChannelPublisher(ARM_COMMAND_TOPIC, LowCmd_)
        publisher.Init()
        subscriber = ChannelSubscriber(LOW_STATE_TOPIC, LowState_)
        subscriber.Init(self._low_state_callback, 10)
        motion_client = MotionSwitcherClient()
        motion_client.SetTimeout(1.0)
        motion_client.Init()
        self._publisher = publisher
        self._subscriber = subscriber
        # Do not subscribe to rt/arm_sdk here.  The stock G1 can have another
        # arm producer with a firmware-specific DDS layout; decoding that
        # foreign command frame crashes the SDK process.  XR Teleoperate uses
        # the same safe pattern: publish arm_sdk and subscribe only to
        # rt/lowstate.
        self._arm_subscriber = None
        self._motion_client = motion_client
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._started_at = self._monotonic()
        self._ownership_conflict = False
        self._published_fingerprints.clear()
        self._motion_mode_name = None
        self._motion_mode_checked = False
        self._motion_mode_verified_at = None
        self._motion_mode_error = None
        self._motion_stop.clear()
        self._started = True
        self._motion_thread = threading.Thread(
            target=self._poll_motion_mode,
            daemon=True,
            name="unitree-native-motion-mode-monitor",
        )
        self._motion_thread.start()

    def latest_state(self) -> Optional[RobotState]:
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
                and self._motion_mode_verified_at is not None
                and now - self._motion_mode_verified_at <= self.motion_mode_grace_s
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
            raise RuntimeError("Native Unitree arm hardware has not been started")
        low_cmd = self._low_cmd
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = command.mode_machine
        # Unitree's rt/arm_sdk consumer expects a complete G1 command frame.
        # Keep every non-arm joint at its measured position so the packet is
        # valid without taking ownership of the legs or waist.
        with self._lock:
            state = self._state
        if state is not None and len(state.body_q) >= 29:
            for joint_index in range(29):
                motor = low_cmd.motor_cmd[joint_index]
                motor.mode = 1
                motor.q = state.body_q[joint_index]
                motor.dq = 0.0
                motor.tau = 0.0
                if joint_index in _WRIST_INDICES:
                    motor.kp = 40.0
                    motor.kd = 1.5
                elif joint_index in _WEAK_INDICES:
                    motor.kp = 80.0
                    motor.kd = 3.0
                else:
                    motor.kp = 300.0
                    motor.kd = 3.0
        for offset, joint_index in enumerate(ARM_INDICES):
            motor = low_cmd.motor_cmd[joint_index]
            motor.mode = 1
            motor.q = command.q[offset]
            motor.dq = command.dq[offset]
            motor.kp = command.kp[offset]
            motor.kd = command.kd[offset]
            motor.tau = 0.0
        # Unitree's arm_sdk convention uses the otherwise-unused motor slot
        # as an arm arbitration weight.  Waist and legs remain untouched.
        low_cmd.motor_cmd[ARM_WEIGHT_INDEX].q = command.weight
        low_cmd.crc = self._crc.Crc(low_cmd)
        fingerprint = tuple(command.q) + (float(command.weight), float(command.mode_machine))
        with self._lock:
            self._published_fingerprints.append(fingerprint)
        write = getattr(self._publisher, "Write", None)
        if callable(write):
            write(low_cmd)
        else:  # pragma: no cover - compatibility with minimal test doubles
            publish = getattr(self._publisher, "publish", None)
            if not callable(publish):
                raise RuntimeError("native Unitree publisher has no Write method")
            publish(low_cmd)

    def close(self) -> None:
        # The safety controller publishes its bounded zero-weight ramp before
        # closing this transport.
        if not self._started:
            return
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
        self._started_at = None

    def _poll_motion_mode(self) -> None:
        while not self._motion_stop.is_set():
            client = self._motion_client
            try:
                result = client.CheckMode() if client is not None else None
                name = None
                if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
                    candidate = result[1].get("name")
                    name = str(candidate).strip() if candidate is not None else None
                if not name:
                    raise RuntimeError("MotionSwitcherClient.CheckMode returned no mode name")
                with self._lock:
                    # A successful reply that names a *different* mode must
                    # take effect immediately.  A transient RPC failure below
                    # must not turn a verified \"ai\" mode into a false fault.
                    self._motion_mode_name = name
                    self._motion_mode_checked = True
                    self._motion_mode_verified_at = self._monotonic()
                    self._motion_mode_error = None
            except Exception as exc:
                # CheckMode occasionally drops a reply on the stock G1 image
                # while native arm DDS remains healthy.  Keep the last
                # verified result only for the bounded grace window; a real
                # mode change is still applied immediately on the next valid
                # reply and a persistent outage fails closed after the grace.
                with self._lock:
                    self._motion_mode_error = f"{type(exc).__name__}: {exc}"
            self._motion_stop.wait(self.motion_poll_s)

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
                all(abs(actual - wanted) <= 1e-4 for actual, wanted in zip(fingerprint, expected))
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
            motor_faults = []
            motor_status_verified = False
            for index in range(29):
                motor = message.motor_state[index]
                temperatures = self._temperatures(getattr(motor, "temperature", None))
                if temperatures:
                    motor_status_verified = True
                    for sensor_index, temperature in enumerate(temperatures):
                        if not math.isfinite(temperature) or temperature > 85.0:
                            motor_faults.append(
                                f"motor_{index}_temperature_{sensor_index}"
                                if len(temperatures) > 1
                                else f"motor_{index}_temperature"
                            )
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
                # latest_state() overlays fail-closed mode and ownership checks.
                compatible_motion_mode=False,
                controller_available=False,
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

    @staticmethod
    def _temperatures(value: object) -> tuple[float, ...]:
        """Normalize SDK scalar and ROS ``int16[2]`` motor temperatures."""

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (float(value),)
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return ()
        try:
            return tuple(float(item) for item in value)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return ()
