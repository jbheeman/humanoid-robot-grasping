"""Persistent, fail-closed controller for Unitree's ``rt/arm_sdk`` topic.

The state machine in this module is independent of SDK2 and wall-clock sleeps.
That keeps the safety policy deterministic and permits full testing without a
robot.  The SDK-specific transport is implemented in :mod:`arm_unitree`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Callable, Protocol, Sequence


LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINT_NAMES)
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

# G1 29-DOF limits from the Unitree description, narrowed again by
# ``joint_limit_margin_rad`` before targets are accepted.  The right shoulder
# roll range is mirrored from the left arm.
DEFAULT_RIGHT_JOINT_LIMITS = (
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.9722, 1.9722),
    (-1.6144, 1.6144),
    (-1.6144, 1.6144),
)


class ArmState(str, Enum):
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    ARMED = "ARMED"
    HOLDING = "HOLDING"
    FAULT = "FAULT"


class ArmBridgeError(ValueError):
    """A request was rejected without changing the controller state."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RobotState:
    """The latest robot observation, expressed in bridge joint order."""

    arm_q: tuple[float, ...]
    received_at: float
    standing: bool
    standing_since: float | None
    compatible_motion_mode: bool = True
    controller_available: bool = True
    mode_machine: int = 0
    waist_q: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.arm_q) != 14:
            raise ValueError("RobotState.arm_q must contain all 14 arm joints")
        if len(self.waist_q) != 3:
            raise ValueError("RobotState.waist_q must contain yaw, roll, and pitch")


@dataclass(frozen=True)
class ArmCommand:
    """One complete command for ``rt/arm_sdk`` (never a partial right arm)."""

    q: tuple[float, ...]
    dq: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    weight: float
    mode_machine: int
    published_at: float

    def __post_init__(self) -> None:
        if not all(len(values) == 14 for values in (self.q, self.dq, self.kp, self.kd)):
            raise ValueError("ArmCommand must represent all 14 arm joints")


class ArmHardware(Protocol):
    """Small SDK boundary used by the persistent controller."""

    def start(self) -> None: ...

    def latest_state(self) -> RobotState | None: ...

    def publish(self, command: ArmCommand) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ArmBridgeConfig:
    allow_movement: bool = False
    calibration_id: str | None = None
    control_hz: float = 250.0
    target_ttl_s: float = 0.250
    deadman_s: float = 0.500
    state_ttl_s: float = 0.250
    stable_standing_s: float = 2.0
    weight_ramp_s: float = 0.250
    joint_limit_margin_rad: float = 0.05
    max_target_delta_rad: float = 0.05
    max_velocity_rad_s: float = 0.50
    max_acceleration_rad_s2: float = 2.0
    max_following_error_rad: float = 0.35
    kp: float = 60.0
    kd: float = 1.5
    right_joint_limits: tuple[tuple[float, float], ...] = DEFAULT_RIGHT_JOINT_LIMITS
    waist_reference_rad: tuple[float, float, float] | None = None
    max_waist_deviation_rad: float = math.radians(3.0)

    def __post_init__(self) -> None:
        positive = {
            "control_hz": self.control_hz,
            "target_ttl_s": self.target_ttl_s,
            "deadman_s": self.deadman_s,
            "state_ttl_s": self.state_ttl_s,
            "stable_standing_s": self.stable_standing_s,
            "weight_ramp_s": self.weight_ramp_s,
            "max_target_delta_rad": self.max_target_delta_rad,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "max_acceleration_rad_s2": self.max_acceleration_rad_s2,
            "max_following_error_rad": self.max_following_error_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.control_hz < 50.0 or self.control_hz > 250.0:
            raise ValueError("control_hz must be within the verified 50-250 Hz range")
        if len(self.right_joint_limits) != 7:
            raise ValueError("right_joint_limits must contain seven ranges")
        if self.joint_limit_margin_rad < 0.0:
            raise ValueError("joint_limit_margin_rad must be >= 0")
        if self.waist_reference_rad is not None and len(self.waist_reference_rad) != 3:
            raise ValueError("waist_reference_rad must contain yaw, roll, and pitch")


@dataclass
class LoopMetrics:
    ticks: int = 0
    missed_deadlines: int = 0
    max_lateness_s: float = 0.0
    last_period_s: float | None = None
    max_period_s: float = 0.0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "ticks": self.ticks,
            "missed_deadlines": self.missed_deadlines,
            "max_lateness_ms": round(self.max_lateness_s * 1000.0, 3),
            "last_period_ms": None
            if self.last_period_s is None
            else round(self.last_period_s * 1000.0, 3),
            "max_period_ms": round(self.max_period_s * 1000.0, 3),
        }


class ArmBridgeController:
    """Owns the persistent arm publisher and its safety state machine."""

    def __init__(
        self,
        hardware: ArmHardware,
        config: ArmBridgeConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.hardware = hardware
        self.config = config
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = monotonic()
        self.state = ArmState.DISARMED
        self.session_id: str | None = None
        self.calibration_id: str | None = None
        self.last_sequence = -1
        self.last_target_at: float | None = None
        self.last_target_source_timestamp: float | None = None
        self.fault_reason: str | None = None
        self.hold_reason: str | None = None
        self.left_latch: tuple[float, ...] | None = None
        self.commanded_right: list[float] | None = None
        self.desired_right: tuple[float, ...] | None = None
        self.right_velocity = [0.0] * 7
        self.weight = 0.0
        self._transition_at = self.started_at
        self._release_start_weight = 0.0
        self._last_tick_at: float | None = None
        self._last_publish_at: float | None = None
        self.metrics = LoopMetrics()

    def start(self) -> None:
        """Start the SDK transport and the one persistent control loop."""
        with self._lock:
            if self._thread is not None:
                return
            self.hardware.start()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="unitree-arm-control", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        """Request a release, run its bounded ramp, then close the transport."""
        deadline = self._monotonic() + self.config.weight_ramp_s
        with self._lock:
            self._begin_holding("bridge_shutdown", self._monotonic())
        while self.weight > 0.0 and self._monotonic() < deadline:
            time.sleep(min(1.0 / self.config.control_hz, 0.01))
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self.hardware.close()
        self._thread = None

    def enable(self, *, session_id: object, calibration_id: object) -> dict[str, object]:
        now = self._monotonic()
        session = self._required_string(session_id, "session_id")
        calibration = self._required_string(calibration_id, "calibration_id")
        with self._lock:
            if not self.config.allow_movement:
                raise ArmBridgeError(
                    "Movement is disabled; restart with --allow-movement", code="movement_disabled"
                )
            if not self.config.calibration_id:
                raise ArmBridgeError(
                    "No validated calibration is configured", code="calibration_required"
                )
            if calibration != self.config.calibration_id:
                raise ArmBridgeError(
                    "Calibration ID does not match the validated robot calibration",
                    code="calibration_mismatch",
                )
            if self.state not in (ArmState.DISARMED,):
                raise ArmBridgeError(f"Cannot enable from {self.state.value}", code="invalid_state")
            robot = self._require_safe_robot_state(now, require_stable=True)
            self.session_id = session
            self.calibration_id = calibration
            self.last_sequence = -1
            self.last_target_at = now
            self.last_target_source_timestamp = None
            self.left_latch = tuple(robot.arm_q[:7])
            self.commanded_right = list(robot.arm_q[7:])
            self.desired_right = tuple(robot.arm_q[7:])
            self.right_velocity = [0.0] * 7
            self.weight = 0.0
            self.fault_reason = None
            self.hold_reason = None
            self.state = ArmState.ARMING
            self._transition_at = now
            return self.state_report(now)

    def set_target(
        self,
        *,
        session_id: object,
        sequence: object,
        calibration_id: object,
        right_arm_q: object,
        source_timestamp: object,
    ) -> dict[str, object]:
        now = self._monotonic()
        with self._lock:
            if self.state is not ArmState.ARMED:
                raise ArmBridgeError("Arm must be ARMED before accepting targets", code="not_armed")
            if self._required_string(session_id, "session_id") != self.session_id:
                raise ArmBridgeError(
                    "Target session does not match the enabled session", code="session_mismatch"
                )
            if self._required_string(calibration_id, "calibration_id") != self.calibration_id:
                raise ArmBridgeError(
                    "Target calibration does not match the enabled calibration",
                    code="calibration_mismatch",
                )
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ArmBridgeError("sequence must be an integer", code="invalid_sequence")
            if sequence <= self.last_sequence:
                raise ArmBridgeError("sequence must increase strictly", code="stale_sequence")
            try:
                source = float(source_timestamp)
            except (TypeError, ValueError) as exc:
                raise ArmBridgeError(
                    "source_timestamp must be a Unix timestamp", code="invalid_timestamp"
                ) from exc
            if not math.isfinite(source):
                raise ArmBridgeError("source_timestamp must be finite", code="invalid_timestamp")
            age = self._wall_time() - source
            if age < -0.050 or age > self.config.target_ttl_s:
                ttl_ms = round(self.config.target_ttl_s * 1000.0)
                raise ArmBridgeError(
                    f"Target source timestamp is outside the {ttl_ms} ms TTL", code="stale_target"
                )
            target = self._validate_target(right_arm_q)
            assert self.commanded_right is not None
            if any(
                abs(target[i] - self.commanded_right[i]) > self.config.max_target_delta_rad
                for i in range(7)
            ):
                raise ArmBridgeError(
                    f"Target exceeds the {self.config.max_target_delta_rad:.3f} rad maximum command delta",
                    code="discontinuous_target",
                )
            self._require_safe_robot_state(now, require_stable=False)
            self.desired_right = target
            self.last_sequence = sequence
            self.last_target_at = now
            self.last_target_source_timestamp = source
            return self.state_report(now)

    def stop(self, reason: str = "operator_stop") -> dict[str, object]:
        """Idempotently start the highest-priority bounded release path."""
        now = self._monotonic()
        with self._lock:
            self._begin_holding(reason, now)
            return self.state_report(now)

    def tick(self, now: float | None = None) -> None:
        """Advance one control period; public for deterministic fake-clock tests."""
        if now is None:
            now = self._monotonic()
        with self._lock:
            previous = self._last_tick_at
            dt = 1.0 / self.config.control_hz if previous is None else max(0.0, now - previous)
            self._last_tick_at = now
            self.metrics.ticks += 1
            if previous is not None:
                self.metrics.last_period_s = dt
                self.metrics.max_period_s = max(self.metrics.max_period_s, dt)

            if self.state is ArmState.DISARMED:
                return

            robot = self.hardware.latest_state()
            if robot is None or now - robot.received_at > self.config.state_ttl_s:
                self._enter_fault("robot_state_stale", now)
            elif not self._finite(robot.arm_q):
                self._enter_fault("robot_state_non_finite", now)
            elif not robot.compatible_motion_mode:
                self._enter_fault("incompatible_motion_mode", now)
            elif not robot.controller_available:
                self._enter_fault("competing_arm_controller", now)

            if self.state is ArmState.ARMING:
                ratio = min(1.0, (now - self._transition_at) / self.config.weight_ramp_s)
                self.weight = ratio
                if ratio >= 1.0:
                    self.state = ArmState.ARMED
                    self._transition_at = now

            if self.state is ArmState.ARMED:
                if not robot or not robot.standing:
                    self._enter_fault("standing_state_lost", now)
                elif (
                    self.last_target_at is not None
                    and now - self.last_target_at >= self._deadline_trigger_s(self.config.deadman_s)
                ):
                    self._begin_holding("target_deadman", now)
                elif self.commanded_right is not None and self.weight >= 0.5:
                    measured_right = robot.arm_q[7:]
                    if any(
                        abs(measured_right[i] - self.commanded_right[i])
                        > self.config.max_following_error_rad
                        for i in range(7)
                    ):
                        self._enter_fault("following_error", now)

            if self.state is ArmState.ARMED:
                self._interpolate_right(dt)
            elif self.state in (ArmState.HOLDING, ArmState.FAULT):
                elapsed = max(0.0, now - self._transition_at)
                release_s = self._deadline_trigger_s(self.config.weight_ramp_s)
                self.weight = self._release_start_weight * max(0.0, 1.0 - elapsed / release_s)

            self._publish(now, robot)

            if self.state is ArmState.HOLDING and self.weight <= 0.0:
                self._reset_disarmed(now)

    def health_report(self) -> dict[str, object]:
        with self._lock:
            robot = self.hardware.latest_state()
            robot_state_fresh = (
                robot is not None
                and self._monotonic() - robot.received_at <= self.config.state_ttl_s
            )
            return {
                "ok": self.state is not ArmState.FAULT,
                "bridge": "unitree_arm_bridge",
                "state": self.state.value,
                "allow_movement": self.config.allow_movement,
                "calibration_configured": bool(self.config.calibration_id),
                "robot_state_fresh": robot_state_fresh,
                "control_hz": self.config.control_hz,
                "target_ttl_ms": round(self.config.target_ttl_s * 1000.0),
                "deadman_ms": round(self.config.deadman_s * 1000.0),
                "weight_ramp_ms": round(self.config.weight_ramp_s * 1000.0),
                "uptime_s": round(self._monotonic() - self.started_at, 3),
                "loop": self.metrics.as_dict(),
                "fault_reason": self.fault_reason,
            }

    def state_report(self, now: float | None = None) -> dict[str, object]:
        if now is None:
            now = self._monotonic()
        with self._lock:
            robot = self.hardware.latest_state()
            return {
                "ok": self.state is not ArmState.FAULT,
                "state": self.state.value,
                "session_id": self.session_id,
                "calibration_id": self.calibration_id,
                "last_sequence": self.last_sequence,
                "last_target_age_ms": (
                    None
                    if self.last_target_at is None
                    else round(max(0.0, now - self.last_target_at) * 1000.0, 3)
                ),
                "robot_state_age_ms": (
                    None if robot is None else round(max(0.0, now - robot.received_at) * 1000.0, 3)
                ),
                "standing": None if robot is None else robot.standing,
                "compatible_motion_mode": None if robot is None else robot.compatible_motion_mode,
                "controller_available": None if robot is None else robot.controller_available,
                "weight": round(self.weight, 6),
                "hold_reason": self.hold_reason,
                "fault_reason": self.fault_reason,
                "commanded_arm_q": None
                if self.left_latch is None or self.commanded_right is None
                else [*self.left_latch, *self.commanded_right],
                "loop": self.metrics.as_dict(),
            }

    def _run_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        deadline = self._monotonic()
        while not self._stop_event.is_set():
            now = self._monotonic()
            if now < deadline:
                self._stop_event.wait(deadline - now)
                continue
            lateness = now - deadline
            if lateness > period:
                with self._lock:
                    self.metrics.missed_deadlines += max(1, int(lateness / period))
                    self.metrics.max_lateness_s = max(self.metrics.max_lateness_s, lateness)
                deadline = now
            try:
                self.tick(now)
            except Exception as exc:  # transport errors must release and latch a fault
                with self._lock:
                    self._enter_fault(f"control_loop_error:{type(exc).__name__}", now)
            deadline += period

    def _require_safe_robot_state(self, now: float, *, require_stable: bool) -> RobotState:
        robot = self.hardware.latest_state()
        if robot is None or now - robot.received_at > self.config.state_ttl_s:
            raise ArmBridgeError("Fresh LowState is required", code="robot_state_stale")
        if not self._finite(robot.arm_q):
            raise ArmBridgeError(
                "LowState contains a non-finite arm position", code="robot_state_non_finite"
            )
        if not robot.standing:
            raise ArmBridgeError("Robot is not in a balanced standing state", code="not_standing")
        if require_stable and (
            robot.standing_since is None
            or now - robot.standing_since < self.config.stable_standing_s
        ):
            raise ArmBridgeError(
                f"Standing state must remain stable for {self.config.stable_standing_s:.1f} seconds",
                code="standing_not_stable",
            )
        if not robot.compatible_motion_mode:
            raise ArmBridgeError(
                "Robot motion mode is incompatible with arm_sdk", code="incompatible_motion_mode"
            )
        if not robot.controller_available:
            raise ArmBridgeError(
                "Another arm controller is active", code="competing_arm_controller"
            )
        if self.config.waist_reference_rad is not None:
            if not self._finite(robot.waist_q):
                raise ArmBridgeError(
                    "LowState contains a non-finite waist position", code="waist_state_non_finite"
                )
            if any(
                abs(actual - expected) > self.config.max_waist_deviation_rad
                for actual, expected in zip(
                    robot.waist_q, self.config.waist_reference_rad, strict=True
                )
            ):
                raise ArmBridgeError(
                    "Waist differs by more than 3 degrees from calibration",
                    code="waist_calibration_mismatch",
                )
        return robot

    def _validate_target(self, value: object) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 7:
            raise ArmBridgeError(
                "right_arm_q must contain seven joint positions", code="invalid_target"
            )
        try:
            target = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ArmBridgeError(
                "right_arm_q must contain numeric positions", code="invalid_target"
            ) from exc
        if not self._finite(target):
            raise ArmBridgeError("right_arm_q must contain finite positions", code="invalid_target")
        margin = self.config.joint_limit_margin_rad
        for index, (position, limits) in enumerate(
            zip(target, self.config.right_joint_limits, strict=True)
        ):
            lower, upper = limits
            if position < lower + margin or position > upper - margin:
                raise ArmBridgeError(
                    f"{RIGHT_ARM_JOINT_NAMES[index]} violates its limit margin",
                    code="joint_limit",
                )
        return target

    def _interpolate_right(self, dt: float) -> None:
        if self.commanded_right is None or self.desired_right is None or dt <= 0.0:
            return
        vmax = self.config.max_velocity_rad_s
        amax = self.config.max_acceleration_rad_s2
        for index in range(7):
            error = self.desired_right[index] - self.commanded_right[index]
            requested_velocity = max(-vmax, min(vmax, error / dt))
            previous_velocity = self.right_velocity[index]
            velocity_change = max(
                -amax * dt, min(amax * dt, requested_velocity - previous_velocity)
            )
            velocity = previous_velocity + velocity_change
            step = velocity * dt
            if abs(step) > abs(error):
                step = error
                velocity = 0.0
            self.commanded_right[index] += step
            self.right_velocity[index] = velocity

    def _publish(self, now: float, robot: RobotState | None) -> None:
        if (
            not self.config.allow_movement
            or self.left_latch is None
            or self.commanded_right is None
        ):
            return
        mode_machine = 0 if robot is None else robot.mode_machine
        q = (*self.left_latch, *self.commanded_right)
        command = ArmCommand(
            q=q,
            dq=(0.0,) * 14,
            kp=(self.config.kp,) * 14,
            kd=(self.config.kd,) * 14,
            weight=max(0.0, min(1.0, self.weight)),
            mode_machine=mode_machine,
            published_at=now,
        )
        self.hardware.publish(command)
        self._last_publish_at = now

    def _begin_holding(self, reason: str, now: float) -> None:
        if self.state in (ArmState.DISARMED, ArmState.FAULT):
            return
        if self.state is ArmState.HOLDING:
            return
        self.state = ArmState.HOLDING
        self.hold_reason = reason
        self._release_start_weight = self.weight
        self._transition_at = now
        self.desired_right = None if self.commanded_right is None else tuple(self.commanded_right)
        self.right_velocity = [0.0] * 7

    def _enter_fault(self, reason: str, now: float) -> None:
        if self.state is ArmState.FAULT:
            return
        self.state = ArmState.FAULT
        self.fault_reason = reason
        self._release_start_weight = self.weight
        self._transition_at = now
        self.desired_right = None if self.commanded_right is None else tuple(self.commanded_right)
        self.right_velocity = [0.0] * 7

    def _reset_disarmed(self, now: float) -> None:
        self.state = ArmState.DISARMED
        self.session_id = None
        self.calibration_id = None
        self.last_sequence = -1
        self.last_target_at = None
        self.last_target_source_timestamp = None
        self.left_latch = None
        self.commanded_right = None
        self.desired_right = None
        self.right_velocity = [0.0] * 7
        self.weight = 0.0
        self._transition_at = now

    @staticmethod
    def _required_string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ArmBridgeError(f"{name} must be a non-empty string", code="invalid_request")
        return value.strip()

    @staticmethod
    def _finite(values: Sequence[float]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _deadline_trigger_s(self, maximum_s: float) -> float:
        """Reserve one control period so an observed action meets its maximum."""
        period = 1.0 / self.config.control_hz
        return max(period, maximum_s - period)
