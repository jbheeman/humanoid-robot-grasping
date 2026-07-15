"""Simple dual-arm manual controller for the robot-local Unitree SDK2 edge.

This controller deliberately has no camera, calibration, IK, or browser
commissioning dependency. ROS commands are absolute joint targets, while this
class remains the sole publisher of complete 14-joint ``rt/arm_sdk`` frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import secrets
import threading
import time
from typing import Callable, Optional, Sequence

from .arm_bridge import ArmBridgeError, ArmCommand, ArmHardware, ArmState, RobotState
from .joints import (
    DEFAULT_LEFT_JOINT_LIMITS,
    DEFAULT_RIGHT_JOINT_LIMITS,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)


@dataclass(frozen=True)
class ManualArmConfig:
    allow_movement: bool = False
    control_hz: float = 100.0
    state_ttl_s: float = 0.250
    command_ttl_s: float = 0.250
    heartbeat_ttl_s: float = 0.500
    stable_standing_s: float = 2.0
    weight_ramp_s: float = 0.500
    max_target_delta_rad: float = 0.05
    max_velocity_rad_s: float = 0.25
    max_acceleration_rad_s2: float = 1.0
    max_following_error_rad: float = 0.03
    joint_limit_margin_rad: float = 0.05
    kp: float = 60.0
    kd: float = 1.5
    gain_profile: str = "sdk2"

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "state_ttl_s",
            "command_ttl_s",
            "heartbeat_ttl_s",
            "stable_standing_s",
            "weight_ramp_s",
            "max_target_delta_rad",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
            "max_following_error_rad",
            "kp",
            "kd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 50.0 <= self.control_hz <= 250.0:
            raise ValueError("control_hz must be between 50 and 250 Hz")
        if self.joint_limit_margin_rad < 0.0:
            raise ValueError("joint_limit_margin_rad must be >= 0")
        if self.gain_profile not in {"sdk2", "xr"}:
            raise ValueError("gain_profile must be 'sdk2' or 'xr'")


@dataclass(frozen=True)
class _Trajectory:
    start_q: tuple[float, ...]
    target_q: tuple[float, ...]
    started_at: float
    duration_s: float


def _smoothstep(value: float) -> float:
    bounded = min(max(value, 0.0), 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)


class ManualArmController:
    """Thread-safe, deadman-controlled dual-arm joint-space controller."""

    def __init__(
        self,
        hardware: ArmHardware,
        config: ManualArmConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.hardware = hardware
        self.config = config
        self._monotonic = monotonic
        self._wall_time_ns = wall_time_ns
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = ArmState.DISARMED
        self._session_id: Optional[str] = None
        self._last_heartbeat: Optional[float] = None
        self._last_sequences = {"left": -1, "right": -1}
        self._last_target_at: Optional[float] = None
        self._desired_q: Optional[tuple[float, ...]] = None
        self._commanded_q: Optional[tuple[float, ...]] = None
        self._trajectories: dict[str, Optional[_Trajectory]] = {
            "left": None,
            "right": None,
        }
        self._weight = 0.0
        self._ramp_started_at: Optional[float] = None
        self._release_start_weight = 0.0
        self._release_reason: Optional[str] = None
        self._release_terminal = ArmState.DISARMED
        self._fault_reason: Optional[str] = None
        self._last_rejection: Optional[dict[str, str]] = None
        self._baseline_q: Optional[tuple[float, ...]] = None
        self._maximum_measured_displacement = [0.0] * 14
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.hardware.start()
        self._stop_event.clear()
        self._started = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="g1-manual-arm-controller"
        )
        self._thread.start()

    def close(self) -> None:
        if not self._started:
            return
        self.stop("bridge_shutdown")
        deadline = self._monotonic() + self.config.weight_ramp_s + 0.25
        while self._monotonic() < deadline:
            with self._lock:
                if self._weight <= 0.0:
                    break
            time.sleep(0.01)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self.hardware.close()
        self._thread = None
        self._started = False

    def _gate_failures(self, robot: Optional[RobotState], now: float) -> list[str]:
        if robot is None:
            return ["lowstate_missing"]
        failures: list[str] = []
        if now - robot.received_at > self.config.state_ttl_s:
            failures.append("lowstate_stale")
        if not robot.standing:
            failures.extend(robot.balance_details or ("not_stably_standing",))
        if robot.standing_since is None or now - robot.standing_since < self.config.stable_standing_s:
            failures.append("standing_not_stable")
        if not robot.motion_mode_verified:
            failures.append(f"motion_mode_unverified:{robot.motion_mode_name!r}")
        if not robot.controller_ownership_verified:
            failures.append("arm_sdk_ownership_unverified")
        if not robot.motor_status_verified:
            failures.append("motor_status_unverified")
        if not robot.motor_state_healthy:
            failures.extend(robot.motor_faults or ("motor_state_unhealthy",))
        if any(not math.isfinite(value) for value in (*robot.arm_q, *robot.arm_dq)):
            failures.append("arm_state_non_finite")
        if robot.arm_dq and max(abs(value) for value in robot.arm_dq) > 0.25:
            failures.append("arm_velocity_too_high")
        return list(dict.fromkeys(failures))

    def enable(self, session_id: str = "") -> dict[str, object]:
        with self._lock:
            if not self.config.allow_movement:
                raise ArmBridgeError(
                    "Manual movement is disabled on the robot launcher",
                    code="movement_disabled",
                )
            if self._state is not ArmState.DISARMED:
                raise ArmBridgeError(
                    f"Manual arm cannot enable while {self._state.value}", code="busy"
                )
            now = self._monotonic()
            robot = self.hardware.latest_state()
            failures = self._gate_failures(robot, now)
            if failures:
                raise ArmBridgeError(
                    "Arm preflight failed: " + ", ".join(failures), code="preflight_failed"
                )
            assert robot is not None
            self._session_id = session_id.strip() or secrets.token_urlsafe(18)
            self._last_heartbeat = now
            self._last_sequences = {"left": -1, "right": -1}
            self._last_target_at = None
            self._desired_q = tuple(robot.arm_q)
            self._commanded_q = tuple(robot.arm_q)
            self._baseline_q = tuple(robot.arm_q)
            self._maximum_measured_displacement = [0.0] * 14
            self._trajectories = {"left": None, "right": None}
            self._weight = 0.0
            self._ramp_started_at = now
            self._release_reason = None
            self._fault_reason = None
            self._last_rejection = None
            self._state = ArmState.ARMING
            return self.state_report()

    def heartbeat(self, session_id: str) -> dict[str, object]:
        with self._lock:
            self._require_session(session_id, allow_arming=True)
            self._last_heartbeat = self._monotonic()
            return self.state_report()

    def stop(self, reason: str = "operator_stop") -> dict[str, object]:
        with self._lock:
            if self._state in (ArmState.DISARMED, ArmState.FAULT) and self._weight <= 0.0:
                # Stop is also the explicit, motion-free fault reset. This
                # avoids a stale FAULT surviving forever across operator tries.
                self._state = ArmState.DISARMED
                self._session_id = None
                self._fault_reason = None
                self._release_reason = str(reason)
                return self.state_report()
            self._begin_release(str(reason), ArmState.DISARMED)
            return self.state_report()

    def _require_session(self, session_id: str, *, allow_arming: bool = False) -> None:
        allowed = (ArmState.ARMED, ArmState.ARMING) if allow_arming else (ArmState.ARMED,)
        if self._state not in allowed or not self._session_id:
            raise ArmBridgeError(
                f"Manual arm is {self._state.value} and no session is armed", code="not_armed"
            )
        if str(session_id) != self._session_id:
            raise ArmBridgeError("Manual arm session does not match", code="session_mismatch")

    def set_side_target(
        self,
        *,
        side: str,
        session_id: str,
        sequence: int,
        joint_names: Sequence[str],
        position_rad: Sequence[float],
        duration_s: float,
        sent_time_ns: int,
    ) -> dict[str, object]:
        side = str(side).strip().lower()
        expected_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        limits = DEFAULT_LEFT_JOINT_LIMITS if side == "left" else DEFAULT_RIGHT_JOINT_LIMITS
        if side not in ("left", "right"):
            raise ArmBridgeError("side must be left or right", code="invalid_side")
        names = tuple(str(value) for value in joint_names)
        positions = tuple(float(value) for value in position_rad)
        if names != expected_names:
            raise ArmBridgeError(
                f"{side} joint_names do not match the canonical G1 order",
                code="joint_contract_mismatch",
            )
        if len(positions) != 7 or any(not math.isfinite(value) for value in positions):
            raise ArmBridgeError("position_rad must contain seven finite values")
        if not math.isfinite(duration_s) or not 0.1 <= duration_s <= 10.0:
            raise ArmBridgeError("move_duration must be between 0.1 and 10 seconds")
        age_s = (self._wall_time_ns() - int(sent_time_ns)) / 1e9
        if sent_time_ns <= 0 or age_s > self.config.command_ttl_s or age_s < -1.0:
            raise ArmBridgeError(
                f"Command timestamp is not fresh (age={age_s:.3f}s)", code="stale_command"
            )
        with self._lock:
            self._require_session(session_id)
            if int(sequence) <= self._last_sequences[side]:
                raise ArmBridgeError("Command sequence must increase", code="stale_sequence")
            assert self._desired_q is not None
            offset = 0 if side == "left" else 7
            current = self._desired_q[offset : offset + 7]
            deltas = tuple(abs(target - start) for target, start in zip(positions, current))
            if max(deltas, default=0.0) > self.config.max_target_delta_rad + 1e-9:
                raise ArmBridgeError(
                    f"Target step exceeds {self.config.max_target_delta_rad:.3f} rad",
                    code="target_step_too_large",
                )
            for name, value, (lower, upper) in zip(names, positions, limits):
                margin = self.config.joint_limit_margin_rad
                if not lower + margin <= value <= upper - margin:
                    raise ArmBridgeError(
                        f"{name} target {value:.4f} is outside the guarded limit",
                        code="joint_limit",
                    )
            maximum_delta = max(deltas, default=0.0)
            minimum_duration = max(
                1.5 * maximum_delta / self.config.max_velocity_rad_s,
                math.sqrt(6.0 * maximum_delta / self.config.max_acceleration_rad_s2),
            )
            if duration_s + 1e-9 < minimum_duration:
                raise ArmBridgeError(
                    f"move_duration must be at least {minimum_duration:.3f}s for this step",
                    code="duration_too_short",
                )
            target = list(self._desired_q)
            target[offset : offset + 7] = positions
            now = self._monotonic()
            start_q = self._commanded_q or self._desired_q
            self._desired_q = tuple(target)
            self._trajectories[side] = _Trajectory(
                tuple(start_q[offset : offset + 7]),
                tuple(positions),
                now,
                duration_s,
            )
            self._last_sequences[side] = int(sequence)
            self._last_target_at = now
            self._last_rejection = None
            return self.state_report()

    def record_rejection(self, side: str, exc: Exception) -> None:
        with self._lock:
            self._last_rejection = {
                "side": str(side),
                "code": str(getattr(exc, "code", "invalid_request")),
                "message": str(exc),
            }

    def _begin_release(self, reason: str, terminal: ArmState) -> None:
        if self._state is ArmState.HOLDING and self._release_terminal is ArmState.FAULT:
            return
        self._state = ArmState.HOLDING
        self._release_reason = reason
        self._release_terminal = terminal
        self._ramp_started_at = self._monotonic()
        self._release_start_weight = self._weight
        self._trajectories = {"left": None, "right": None}

    def _fault(self, reason: str) -> None:
        self._fault_reason = reason
        self._begin_release(reason, ArmState.FAULT)

    def _command(self, robot: RobotState, now: float) -> ArmCommand:
        q = self._commanded_q or tuple(robot.arm_q)
        if self.config.gain_profile == "xr":
            # Current Unitree xr_teleoperate G1 profile.
            side_kp = (80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0)
            side_kd = (3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5)
        else:
            # Installed official SDK2 G1 arm7 example defaults.
            side_kp = (self.config.kp,) * 7
            side_kd = (self.config.kd,) * 7
        return ArmCommand(
            q=tuple(q),
            dq=(0.0,) * 14,
            kp=side_kp + side_kp,
            kd=side_kd + side_kd,
            weight=self._weight,
            mode_machine=robot.mode_machine,
            published_at=now,
        )

    def tick(self) -> None:
        with self._lock:
            now = self._monotonic()
            robot = self.hardware.latest_state()
            if self._state in (ArmState.ARMING, ArmState.ARMED):
                failures = self._gate_failures(robot, now)
                if failures:
                    self._fault("runtime_gate:" + ",".join(failures))
                elif self._last_heartbeat is None or now - self._last_heartbeat > self.config.heartbeat_ttl_s:
                    self._begin_release("heartbeat_timeout", ArmState.DISARMED)
            if robot is None:
                return
            if self._baseline_q is not None:
                self._maximum_measured_displacement = [
                    max(previous, abs(actual - baseline))
                    for previous, actual, baseline in zip(
                        self._maximum_measured_displacement,
                        robot.arm_q,
                        self._baseline_q,
                    )
                ]
            if self._state is ArmState.ARMING:
                assert self._ramp_started_at is not None
                ratio = (now - self._ramp_started_at) / self.config.weight_ramp_s
                self._weight = _smoothstep(ratio)
                if ratio >= 1.0:
                    self._weight = 1.0
                    self._state = ArmState.ARMED
            elif self._state is ArmState.ARMED and self._commanded_q is not None:
                commanded = list(self._commanded_q)
                for side, offset in (("left", 0), ("right", 7)):
                    trajectory = self._trajectories[side]
                    if trajectory is None:
                        continue
                    ratio = (now - trajectory.started_at) / trajectory.duration_s
                    blend = _smoothstep(ratio)
                    commanded[offset : offset + 7] = [
                        (1.0 - blend) * start + blend * target
                        for start, target in zip(trajectory.start_q, trajectory.target_q)
                    ]
                    if ratio >= 1.0:
                        commanded[offset : offset + 7] = trajectory.target_q
                        self._trajectories[side] = None
                self._commanded_q = tuple(commanded)
            elif self._state is ArmState.HOLDING:
                assert self._ramp_started_at is not None
                ratio = min((now - self._ramp_started_at) / self.config.weight_ramp_s, 1.0)
                self._weight = self._release_start_weight * (1.0 - _smoothstep(ratio))
                if ratio >= 1.0:
                    self._weight = 0.0
                    self._state = self._release_terminal
                    self._session_id = None
                    self._last_heartbeat = None
            if self._state in (ArmState.ARMING, ArmState.ARMED, ArmState.HOLDING):
                if self._commanded_q is not None:
                    following = max(
                        abs(actual - commanded)
                        for actual, commanded in zip(robot.arm_q, self._commanded_q)
                    )
                    if following > self.config.max_following_error_rad:
                        self._fault(f"following_error:{following:.4f}")
                self.hardware.publish(self._command(robot, now))

    def _run_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        while not self._stop_event.is_set():
            started = self._monotonic()
            try:
                self.tick()
            except Exception as exc:  # hardware or policy failure must fail closed
                with self._lock:
                    self._fault(f"controller_exception:{type(exc).__name__}:{exc}")
            self._stop_event.wait(max(0.0, period - (self._monotonic() - started)))

    def state_report(self) -> dict[str, object]:
        with self._lock:
            now = self._monotonic()
            robot = self.hardware.latest_state()
            age_ms = None if robot is None else max(0.0, (now - robot.received_at) * 1000.0)
            heartbeat_age_ms = (
                None
                if self._last_heartbeat is None
                else max(0.0, (now - self._last_heartbeat) * 1000.0)
            )
            target_age_ms = (
                None
                if self._last_target_at is None
                else max(0.0, (now - self._last_target_at) * 1000.0)
            )
            return {
                "ok": self._state not in (ArmState.FAULT,),
                "state": self._state.value,
                "session_id": self._session_id,
                "control_mode": "manual",
                "gain_profile": self.config.gain_profile,
                "control_hz": self.config.control_hz,
                "calibration_id": None,
                "last_sequence": max(self._last_sequences.values()),
                "last_sequences": dict(self._last_sequences),
                "last_target_age_ms": target_age_ms,
                "heartbeat_age_ms": heartbeat_age_ms,
                "robot_state_age_ms": age_ms,
                "measured_arm_q": None if robot is None else list(robot.arm_q),
                "measured_arm_dq": None if robot is None else list(robot.arm_dq),
                "commanded_arm_q": None if self._commanded_q is None else list(self._commanded_q),
                "desired_arm_q": None if self._desired_q is None else list(self._desired_q),
                "baseline_arm_q": None if self._baseline_q is None else list(self._baseline_q),
                "maximum_measured_displacement_rad": list(
                    self._maximum_measured_displacement
                ),
                "standing": False if robot is None else robot.standing,
                "compatible_motion_mode": False if robot is None else robot.compatible_motion_mode,
                "controller_available": False if robot is None else robot.controller_available,
                "motion_mode_name": None if robot is None else robot.motion_mode_name,
                "motion_mode_verified": False if robot is None else robot.motion_mode_verified,
                "controller_ownership_verified": (
                    False if robot is None else robot.controller_ownership_verified
                ),
                "motor_status_verified": False if robot is None else robot.motor_status_verified,
                "motor_state_healthy": False if robot is None else robot.motor_state_healthy,
                "motor_faults": [] if robot is None else list(robot.motor_faults),
                "weight": self._weight,
                "fault_reason": self._fault_reason,
                "hold_reason": self._release_reason,
                "last_rejection": self._last_rejection,
                "allow_movement": self.config.allow_movement,
            }
