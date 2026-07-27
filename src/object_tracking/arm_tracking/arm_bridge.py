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

from ruckig import InputParameter, OutputParameter, Result, Ruckig

from .joints import (
    BODY_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_LIMITS,
    RIGHT_ARM_JOINT_NAMES,
)
from .visualization import visualization_state


_WRIST_ARM_OFFSETS = frozenset((4, 5, 6, 11, 12, 13))


class ArmState(str, Enum):
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    ARMED = "ARMED"
    HOLDING = "HOLDING"
    FAULT = "FAULT"


class ArmControlMode(str, Enum):
    TRACKING = "tracking"
    COMMISSIONING = "commissioning"


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
    arm_dq: tuple[float, ...] = (0.0,) * 14
    motion_mode_name: str | None = None
    motion_mode_verified: bool = False
    controller_ownership_verified: bool = False
    motor_status_verified: bool = False
    motor_state_healthy: bool = False
    motor_faults: tuple[str, ...] = ()
    balance_details: tuple[str, ...] = ()
    body_q: tuple[float, ...] = ()
    body_dq: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.arm_q) != 14:
            raise ValueError("RobotState.arm_q must contain all 14 arm joints")
        if len(self.waist_q) != 3:
            raise ValueError("RobotState.waist_q must contain yaw, roll, and pitch")
        if len(self.arm_dq) != 14:
            raise ValueError("RobotState.arm_dq must contain all 14 arm joints")
        if not self.body_q:
            object.__setattr__(self, "body_q", (0.0,) * 12 + self.waist_q + self.arm_q)
        if not self.body_dq:
            object.__setattr__(self, "body_dq", (0.0,) * 15 + self.arm_dq)
        if len(self.body_q) != 29 or len(self.body_dq) != 29:
            raise ValueError("RobotState body_q and body_dq must contain all 29 body joints")


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
    control_mode: ArmControlMode = ArmControlMode.TRACKING
    allow_movement: bool = False
    calibration_id: str | None = None
    joint_contract_id: str | None = None
    control_hz: float = 250.0
    target_ttl_s: float = 0.250
    deadman_s: float = 0.750
    state_ttl_s: float = 0.250
    stable_standing_s: float = 2.0
    standing_loss_grace_s: float = 0.200
    startup_settle_s: float = 0.150
    startup_settle_timeout_s: float = 2.0
    startup_max_velocity_rad_s: float = 0.08
    startup_max_pose_error_rad: float = 0.02
    weight_ramp_s: float = 0.250
    joint_limit_margin_rad: float = 0.05
    max_target_delta_rad: float = 0.05
    max_velocity_rad_s: float = 0.50
    max_acceleration_rad_s2: float = 2.0
    max_jerk_rad_s3: float = 20.0
    max_following_error_rad: float = 0.35
    max_left_drift_rad: float = 0.01
    kp: float = 80.0
    kd: float = 3.0
    right_joint_limits: tuple[tuple[float, float], ...] = DEFAULT_RIGHT_JOINT_LIMITS
    waist_reference_rad: tuple[float, float, float] | None = None
    max_waist_deviation_rad: float = math.radians(3.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_mode", ArmControlMode(self.control_mode))
        positive = {
            "control_hz": self.control_hz,
            "target_ttl_s": self.target_ttl_s,
            "deadman_s": self.deadman_s,
            "state_ttl_s": self.state_ttl_s,
            "stable_standing_s": self.stable_standing_s,
            "standing_loss_grace_s": self.standing_loss_grace_s,
            "startup_settle_timeout_s": self.startup_settle_timeout_s,
            "startup_max_velocity_rad_s": self.startup_max_velocity_rad_s,
            "startup_max_pose_error_rad": self.startup_max_pose_error_rad,
            "weight_ramp_s": self.weight_ramp_s,
            "max_target_delta_rad": self.max_target_delta_rad,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "max_acceleration_rad_s2": self.max_acceleration_rad_s2,
            "max_jerk_rad_s3": self.max_jerk_rad_s3,
            "max_following_error_rad": self.max_following_error_rad,
            "max_left_drift_rad": self.max_left_drift_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not math.isfinite(self.startup_settle_s) or self.startup_settle_s < 0.0:
            raise ValueError("startup_settle_s must be finite and >= 0")
        if self.startup_settle_timeout_s < self.startup_settle_s:
            raise ValueError("startup_settle_timeout_s must cover startup_settle_s")
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
    maximum_command_step_rad: float = 0.0
    maximum_following_error_rad: float = 0.0
    deferred_arming_targets: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "ticks": self.ticks,
            "missed_deadlines": self.missed_deadlines,
            "max_lateness_ms": round(self.max_lateness_s * 1000.0, 3),
            "last_period_ms": None
            if self.last_period_s is None
            else round(self.last_period_s * 1000.0, 3),
            "max_period_ms": round(self.max_period_s * 1000.0, 3),
            "maximum_command_step_rad": round(self.maximum_command_step_rad, 7),
            "maximum_following_error_rad": round(self.maximum_following_error_rad, 7),
            "deferred_arming_targets": self.deferred_arming_targets,
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
        self.fault_details: dict[str, object] | None = None
        self.hold_reason: str | None = None
        self.left_latch: tuple[float, ...] | None = None
        self.commanded_right: list[float] | None = None
        self.desired_right: tuple[float, ...] | None = None
        self.right_velocity = [0.0] * 7
        self.right_acceleration = [0.0] * 7
        self._ruckig = Ruckig(7, 1.0 / self.config.control_hz)
        self._ruckig_input = InputParameter(7)
        self._ruckig_output = OutputParameter(7)
        self.weight = 0.0
        self._transition_at = self.started_at
        self._release_start_weight = 0.0
        self._last_tick_at: float | None = None
        self._last_publish_at: float | None = None
        self._settle_started_at: float | None = None
        self._settle_stable_since: float | None = None
        self._standing_lost_since: float | None = None
        self._arming_phase: str | None = None
        self.metrics = LoopMetrics()

    @property
    def control_mode(self) -> ArmControlMode:
        return self.config.control_mode

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
            if self.control_mode is not ArmControlMode.TRACKING:
                raise ArmBridgeError(
                    "Tracking commands are unavailable in commissioning mode",
                    code="mode_mismatch",
                )
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
            self.right_acceleration = [0.0] * 7
            self._reset_ruckig()
            self.weight = 0.0
            self.fault_reason = None
            self.fault_details = None
            self.hold_reason = None
            self.state = ArmState.ARMING
            self._transition_at = now
            self._settle_started_at = now
            self._settle_stable_since = now
            self._standing_lost_since = None
            self._arming_phase = "settling"
            return self.state_report(now)

    def enable_commissioning(
        self, *, session_id: object, joint_contract_id: object
    ) -> dict[str, object]:
        """Enable the bridge from measured state without camera calibration."""

        now = self._monotonic()
        session = self._required_string(session_id, "session_id")
        contract = self._required_string(joint_contract_id, "joint_contract_id")
        with self._lock:
            if self.control_mode is not ArmControlMode.COMMISSIONING:
                raise ArmBridgeError(
                    "Commissioning commands are unavailable in tracking mode",
                    code="mode_mismatch",
                )
            if not self.config.allow_movement:
                raise ArmBridgeError(
                    "Movement is disabled; restart with --allow-movement",
                    code="movement_disabled",
                )
            if not self.config.joint_contract_id:
                raise ArmBridgeError(
                    "No audited 29-DOF joint contract is configured",
                    code="joint_contract_required",
                )
            if contract != self.config.joint_contract_id:
                raise ArmBridgeError(
                    "Joint contract does not match the audited robot contract",
                    code="joint_contract_mismatch",
                )
            if self.state is not ArmState.DISARMED:
                raise ArmBridgeError(f"Cannot enable from {self.state.value}", code="invalid_state")
            robot = self._require_safe_robot_state(
                now, require_stable=True, require_commissioning_verification=True
            )
            self.session_id = session
            self.calibration_id = None
            self.last_sequence = -1
            self.last_target_at = now
            self.last_target_source_timestamp = None
            self.left_latch = tuple(robot.arm_q[:7])
            self.commanded_right = list(robot.arm_q[7:])
            self.desired_right = tuple(robot.arm_q[7:])
            self.right_velocity = [0.0] * 7
            self.right_acceleration = [0.0] * 7
            self._reset_ruckig()
            self.weight = 0.0
            self.fault_reason = None
            self.fault_details = None
            self.hold_reason = None
            self.state = ArmState.ARMING
            self._transition_at = now
            self._settle_started_at = now
            self._settle_stable_since = now
            self._standing_lost_since = None
            self._arming_phase = "settling"
            return self.state_report(now)

    def set_target(
        self,
        *,
        session_id: object,
        sequence: object,
        calibration_id: object,
        right_arm_q: object,
        source_timestamp: object = None,
        pipeline_age_ms: object = None,
    ) -> dict[str, object]:
        now = self._monotonic()
        with self._lock:
            if self.control_mode is not ArmControlMode.TRACKING:
                raise ArmBridgeError(
                    "Tracking target rejected in commissioning mode", code="mode_mismatch"
                )
            if self.state not in (ArmState.ARMING, ArmState.ARMED):
                raise ArmBridgeError(
                    f"Arm target rejected: bridge state is {self.state.value}; "
                    "create a session and enable at measured pose",
                    code="not_armed",
                )
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
            if pipeline_age_ms is not None:
                if isinstance(pipeline_age_ms, bool) or not isinstance(pipeline_age_ms, int):
                    raise ArmBridgeError(
                        "pipeline_age_ms must be an integer", code="invalid_timestamp"
                    )
                age = float(pipeline_age_ms) / 1000.0
                source = self._wall_time() - age
            else:
                try:
                    source = float(source_timestamp)
                except (TypeError, ValueError) as exc:
                    raise ArmBridgeError(
                        "source_timestamp must be a Unix timestamp", code="invalid_timestamp"
                    ) from exc
                if not math.isfinite(source):
                    raise ArmBridgeError(
                        "source_timestamp must be finite", code="invalid_timestamp"
                    )
                age = self._wall_time() - source
            if age < 0.0 or age > self.config.target_ttl_s:
                ttl_ms = round(self.config.target_ttl_s * 1000.0)
                raise ArmBridgeError(
                    f"Target perception age is outside the {ttl_ms} ms TTL", code="stale_target"
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
            self.last_sequence = sequence
            self.last_target_at = now
            self.last_target_source_timestamp = source
            if self.state is ArmState.ARMING:
                # A perception target may keep the tracking session fresh while
                # the bridge settles, but it must not be queued for release at
                # the end of the weight ramp. The first post-ARMED frame starts
                # a Ruckig trajectory from the verified measured pose.
                self.metrics.deferred_arming_targets += 1
                return self.state_report(now)
            self.desired_right = target
            return self.state_report(now)

    def set_commissioning_target(
        self,
        *,
        session_id: object,
        sequence: object,
        right_arm_q: object,
    ) -> dict[str, object]:
        """Accept one bounded commissioning target using monotonic local freshness."""

        now = self._monotonic()
        with self._lock:
            if self.control_mode is not ArmControlMode.COMMISSIONING:
                raise ArmBridgeError(
                    "Commissioning target rejected in tracking mode", code="mode_mismatch"
                )
            if self.state is not ArmState.ARMED:
                raise ArmBridgeError("Arm must be ARMED before accepting targets", code="not_armed")
            if self._required_string(session_id, "session_id") != self.session_id:
                raise ArmBridgeError(
                    "Target session does not match the enabled session", code="session_mismatch"
                )
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ArmBridgeError("sequence must be an integer", code="invalid_sequence")
            if sequence <= self.last_sequence:
                raise ArmBridgeError("sequence must increase strictly", code="stale_sequence")
            target = self._validate_target(right_arm_q)
            assert self.commanded_right is not None
            if any(
                abs(target[index] - self.commanded_right[index]) > self.config.max_target_delta_rad
                for index in range(7)
            ):
                raise ArmBridgeError(
                    f"Target exceeds the {self.config.max_target_delta_rad:.3f} rad maximum command delta",
                    code="discontinuous_target",
                )
            self._require_safe_robot_state(
                now, require_stable=False, require_commissioning_verification=True
            )
            self.desired_right = target
            self.last_sequence = sequence
            self.last_target_at = now
            return self.state_report(now)

    def heartbeat(self, *, session_id: object) -> dict[str, object]:
        """Validate an active session; only commissioning heartbeats refresh its deadman."""

        now = self._monotonic()
        with self._lock:
            if self.state not in (ArmState.ARMING, ArmState.ARMED):
                raise ArmBridgeError(
                    f"Heartbeat rejected: bridge state is {self.state.value} and no "
                    "session is armed",
                    code="not_armed",
                )
            if self._required_string(session_id, "session_id") != self.session_id:
                raise ArmBridgeError("Heartbeat session mismatch", code="session_mismatch")
            # Tracking freshness comes exclusively from accepted targets. A
            # control heartbeat must never keep stale tracking commands alive.
            if self.control_mode is ArmControlMode.COMMISSIONING:
                self.last_target_at = now
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
            elif (
                self.control_mode is ArmControlMode.COMMISSIONING
                and robot.motor_status_verified
                and not robot.motor_state_healthy
            ):
                self._enter_fault("motor_state_fault", now)

            if self.state is ArmState.ARMING and robot is not None:
                self._advance_arming(now, robot)

            if self.state is ArmState.ARMED:
                if robot and robot.standing:
                    self._standing_lost_since = None
                elif robot:
                    if self._standing_lost_since is None:
                        self._standing_lost_since = now
                        # Hold the last generated pose while the balance signal
                        # is uncertain. If it recovers, Ruckig resumes from
                        # rest; if it persists, the bounded fault release wins.
                        self.right_velocity = [0.0] * 7
                        self.right_acceleration = [0.0] * 7
                    loss_s = now - self._standing_lost_since
                    if loss_s >= self.config.standing_loss_grace_s:
                        self.fault_details = {
                            "duration_ms": round(loss_s * 1000.0, 3),
                            "balance_details": list(robot.balance_details),
                        }
                        self._enter_fault("standing_state_lost", now)
                if self.state is ArmState.ARMED and (
                    self.last_target_at is not None
                    and now - self.last_target_at >= self._deadline_trigger_s(self.config.deadman_s)
                ):
                    self._begin_holding("target_deadman", now)
                elif (
                    self.state is ArmState.ARMED
                    and self.commanded_right is not None
                    and self.weight >= 0.5
                ):
                    measured_right = robot.arm_q[7:]
                    errors = [
                        abs(measured_right[i] - self.commanded_right[i]) for i in range(7)
                    ]
                    self.metrics.maximum_following_error_rad = max(
                        self.metrics.maximum_following_error_rad,
                        max(errors),
                    )
                    worst_joint = max(range(7), key=errors.__getitem__)
                    if errors[worst_joint] > self.config.max_following_error_rad:
                        self.fault_details = {
                            "joint_index": worst_joint,
                            "error_rad": errors[worst_joint],
                            "measured_rad": measured_right[worst_joint],
                            "commanded_rad": self.commanded_right[worst_joint],
                        }
                        self._enter_fault("following_error", now)
                    elif (
                        self.control_mode is ArmControlMode.COMMISSIONING
                        and self.left_latch is not None
                        and any(
                            abs(actual - expected) > self.config.max_left_drift_rad
                            for actual, expected in zip(robot.arm_q[:7], self.left_latch)
                        )
                    ):
                        self._enter_fault("left_arm_drift", now)

            if self.state is ArmState.ARMED and self._standing_lost_since is None:
                try:
                    self._interpolate_right(dt)
                except (ArmBridgeError, RuntimeError, ValueError) as exc:
                    self._enter_fault(f"trajectory_generation:{exc}", now)
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
                "control_mode": self.control_mode.value,
                "state": self.state.value,
                "allow_movement": self.config.allow_movement,
                "calibration_configured": bool(self.config.calibration_id),
                "robot_state_fresh": robot_state_fresh,
                "control_hz": self.config.control_hz,
                "trajectory_generator": "ruckig",
                "max_velocity_rad_s": self.config.max_velocity_rad_s,
                "max_acceleration_rad_s2": self.config.max_acceleration_rad_s2,
                "max_jerk_rad_s3": self.config.max_jerk_rad_s3,
                "target_ttl_ms": round(self.config.target_ttl_s * 1000.0),
                "deadman_ms": round(self.config.deadman_s * 1000.0),
                "weight_ramp_ms": round(self.config.weight_ramp_s * 1000.0),
                "startup_settle_ms": round(self.config.startup_settle_s * 1000.0),
                "standing_loss_grace_ms": round(
                    self.config.standing_loss_grace_s * 1000.0
                ),
                "uptime_s": round(self._monotonic() - self.started_at, 3),
                "loop": self.metrics.as_dict(),
                "fault_reason": self.fault_reason,
                "fault_details": self.fault_details,
            }

    def state_report(self, now: float | None = None) -> dict[str, object]:
        if now is None:
            now = self._monotonic()
        with self._lock:
            robot = self.hardware.latest_state()
            commanded_arm = (
                None
                if self.left_latch is None or self.commanded_right is None
                else [*self.left_latch, *self.commanded_right]
            )
            faulted_joints = []
            if robot is not None:
                for fault in robot.motor_faults:
                    parts = fault.split("_", 2)
                    if len(parts) >= 2 and parts[0] == "motor" and parts[1].isdigit():
                        index = int(parts[1])
                        if index < len(BODY_JOINT_NAMES):
                            faulted_joints.append(BODY_JOINT_NAMES[index])
            visual = visualization_state(
                measured_body_q=None if robot is None else robot.body_q,
                measured_body_dq=None if robot is None else robot.body_dq,
                commanded_arm_q=commanded_arm,
                received_at=None if robot is None else robot.received_at,
                now=now,
                state_ttl_s=self.config.state_ttl_s,
                faulted_joints=faulted_joints,
            )
            return {
                "ok": self.state is not ArmState.FAULT,
                "state": self.state.value,
                "session_id": self.session_id,
                "control_mode": self.control_mode.value,
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
                "motion_mode_name": None if robot is None else robot.motion_mode_name,
                "motion_mode_verified": None if robot is None else robot.motion_mode_verified,
                "controller_ownership_verified": (
                    None if robot is None else robot.controller_ownership_verified
                ),
                "motor_status_verified": None if robot is None else robot.motor_status_verified,
                "motor_state_healthy": None if robot is None else robot.motor_state_healthy,
                "motor_faults": [] if robot is None else list(robot.motor_faults),
                "balance_details": [] if robot is None else list(robot.balance_details),
                "standing_loss_age_ms": (
                    None
                    if self._standing_lost_since is None
                    else round(
                        max(0.0, now - self._standing_lost_since) * 1000.0,
                        3,
                    )
                ),
                "measured_arm_q": None if robot is None else list(robot.arm_q),
                "measured_arm_dq": None if robot is None else list(robot.arm_dq),
                "weight": round(self.weight, 6),
                "arming_phase": self._arming_phase,
                "hold_reason": self.hold_reason,
                "fault_reason": self.fault_reason,
                "fault_details": self.fault_details,
                "commanded_arm_q": commanded_arm,
                "visualization": visual,
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

    def _require_safe_robot_state(
        self,
        now: float,
        *,
        require_stable: bool,
        require_commissioning_verification: bool = False,
    ) -> RobotState:
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
        if require_commissioning_verification and not robot.motion_mode_verified:
            raise ArmBridgeError(
                "Robot motion mode has not been verified for commissioning",
                code="motion_mode_unverified",
            )
        if require_commissioning_verification and not robot.controller_ownership_verified:
            raise ArmBridgeError(
                "Exclusive arm controller ownership has not been verified",
                code="controller_ownership_unverified",
            )
        # Some deployed G1 firmware publishes joint state but no temperature/
        # lost diagnostic fields. Missing fields are not a fault; a diagnostic
        # fault that *is* actually reported remains a hard commissioning gate.
        if (
            require_commissioning_verification
            and robot.motor_status_verified
            and not robot.motor_state_healthy
        ):
            raise ArmBridgeError(
                "Motor status reports a commissioning fault",
                code="motor_state_fault",
            )
        if self.config.waist_reference_rad is not None:
            if not self._finite(robot.waist_q):
                raise ArmBridgeError(
                    "LowState contains a non-finite waist position", code="waist_state_non_finite"
                )
            if any(
                abs(actual - expected) > self.config.max_waist_deviation_rad
                for actual, expected in zip(robot.waist_q, self.config.waist_reference_rad)
            ):
                raise ArmBridgeError(
                    "Waist differs by more than "
                    f"{math.degrees(self.config.max_waist_deviation_rad):.1f} "
                    "degrees from calibration",
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
        for index, (position, limits) in enumerate(zip(target, self.config.right_joint_limits)):
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
        inp = self._ruckig_input
        out = self._ruckig_output
        inp.current_position = list(self.commanded_right)
        inp.current_velocity = list(self.right_velocity)
        inp.current_acceleration = list(self.right_acceleration)
        inp.target_position = list(self.desired_right)
        inp.target_velocity = [0.0] * 7
        inp.target_acceleration = [0.0] * 7
        inp.max_velocity = [self.config.max_velocity_rad_s] * 7
        inp.max_acceleration = [self.config.max_acceleration_rad_s2] * 7
        inp.max_jerk = [self.config.max_jerk_rad_s3] * 7
        if not self._ruckig.validate_input(
            inp,
            check_current_state_within_limits=False,
            check_target_state_within_limits=True,
        ):
            raise ArmBridgeError(
                "Ruckig input violates the configured trajectory limits",
                code="trajectory_validation",
            )
        result = self._ruckig.update(inp, out)
        if result not in (Result.Working, Result.Finished):
            raise ArmBridgeError(
                f"Ruckig rejected the online arm state: {result}",
                code="trajectory_generation",
            )
        position = [float(value) for value in out.new_position]
        velocity = [float(value) for value in out.new_velocity]
        acceleration = [float(value) for value in out.new_acceleration]
        if not self._finite((*position, *velocity, *acceleration)):
            raise ArmBridgeError(
                "Ruckig returned a non-finite arm state",
                code="trajectory_generation",
            )
        previous = tuple(self.commanded_right)
        self.commanded_right[:] = position
        self.right_velocity[:] = velocity
        self.right_acceleration[:] = acceleration
        self.metrics.maximum_command_step_rad = max(
            self.metrics.maximum_command_step_rad,
            max(abs(actual - prior) for actual, prior in zip(position, previous)),
        )
        out.pass_to_input(inp)

    def _advance_arming(self, now: float, robot: RobotState) -> None:
        """Settle at measured pose before allowing the weight ramp to begin."""

        if self.commanded_right is None:
            self._enter_fault("arming_command_unavailable", now)
            return
        if self._arming_phase == "settling":
            measured = robot.arm_q[7:]
            pose_error = max(
                abs(actual - expected)
                for actual, expected in zip(measured, self.commanded_right)
            )
            velocity = max(abs(value) for value in robot.arm_dq[7:])
            stable = (
                pose_error <= self.config.startup_max_pose_error_rad
                and velocity <= self.config.startup_max_velocity_rad_s
            )
            if stable:
                if self._settle_stable_since is None:
                    self._settle_stable_since = now
                if now - self._settle_stable_since >= self.config.startup_settle_s:
                    self._arming_phase = "weight_ramp"
                    self._transition_at = now
            else:
                self._settle_stable_since = None
            started = self._settle_started_at if self._settle_started_at is not None else now
            if now - started > self.config.startup_settle_timeout_s:
                self.fault_details = {
                    "pose_error_rad": pose_error,
                    "velocity_rad_s": velocity,
                }
                self._enter_fault("startup_settle_timeout", now)
            self.weight = 0.0
            return
        ratio = min(1.0, max(0.0, now - self._transition_at) / self.config.weight_ramp_s)
        self.weight = ratio
        if ratio >= 1.0:
            self.state = ArmState.ARMED
            self._transition_at = now
            self._arming_phase = None

    def _reset_ruckig(self) -> None:
        """Start a new online trajectory without retaining a prior session's clock."""

        self._ruckig_input = InputParameter(7)
        self._ruckig_output = OutputParameter(7)

    def _publish(self, now: float, robot: RobotState | None) -> None:
        if (
            not self.config.allow_movement
            or self.left_latch is None
            or self.commanded_right is None
        ):
            return
        mode_machine = 0 if robot is None else robot.mode_machine
        q = (*self.left_latch, *self.commanded_right)
        # Match the gains used by the installed Unitree XR arm controller:
        # shoulder/elbow motors use 80/3 and wrists use 40/1.5.
        kp = tuple(
            40.0 if offset in _WRIST_ARM_OFFSETS else self.config.kp
            for offset in range(14)
        )
        kd = tuple(
            1.5 if offset in _WRIST_ARM_OFFSETS else self.config.kd
            for offset in range(14)
        )
        command = ArmCommand(
            q=q,
            dq=(0.0,) * 14,
            kp=kp,
            kd=kd,
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
        self.right_acceleration = [0.0] * 7

    def _enter_fault(self, reason: str, now: float) -> None:
        if self.state is ArmState.FAULT:
            return
        self.state = ArmState.FAULT
        self.fault_reason = reason
        self._release_start_weight = self.weight
        self._transition_at = now
        self.desired_right = None if self.commanded_right is None else tuple(self.commanded_right)
        self.right_velocity = [0.0] * 7
        self.right_acceleration = [0.0] * 7

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
        self._settle_started_at = None
        self._settle_stable_since = None
        self._standing_lost_since = None
        self._arming_phase = None
        self.right_velocity = [0.0] * 7
        self.right_acceleration = [0.0] * 7
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
