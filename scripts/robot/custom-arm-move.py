#!/usr/bin/env python3
"""One-shot, robot-local G1 right-shoulder SDK2 movement verification."""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import dataclass

from object_tracking.arm_tracking.arm_bridge import ArmCommand, RobotState
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware
from object_tracking.arm_tracking.joints import DEFAULT_RIGHT_JOINT_LIMITS


RIGHT_SHOULDER_PITCH_OFFSET = 7
ARM_JOINT_COUNT = 14


@dataclass
class Observation:
    maximum_selected_displacement_rad: float = 0.0
    maximum_other_drift_rad: float = 0.0

    def update(self, state: RobotState, baseline: tuple[float, ...]) -> None:
        selected = abs(state.arm_q[RIGHT_SHOULDER_PITCH_OFFSET] - baseline[RIGHT_SHOULDER_PITCH_OFFSET])
        other = max(
            abs(actual - start)
            for index, (actual, start) in enumerate(zip(state.arm_q, baseline, strict=True))
            if index != RIGHT_SHOULDER_PITCH_OFFSET
        )
        self.maximum_selected_displacement_rad = max(
            self.maximum_selected_displacement_rad, selected
        )
        self.maximum_other_drift_rad = max(self.maximum_other_drift_rad, other)


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "preflight", "execute"))
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--expected-motion-mode", default="ai")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--weight-ramp-seconds", type=float, default=0.5)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--return-seconds", type=float, default=2.0)
    parser.add_argument("--preflight-timeout", type=float, default=12.0)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.delta <= 0.05:
        raise ValueError("--delta must be > 0 and <= 0.05 rad")
    if not 50.0 <= args.control_hz <= 250.0:
        raise ValueError("--control-hz must be between 50 and 250 Hz")
    for name in (
        "weight_ramp_seconds",
        "ramp_seconds",
        "return_seconds",
        "preflight_timeout",
        "kp",
        "kd",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be finite and >= 0")


def state_failures(state: RobotState | None, now: float) -> list[str]:
    if state is None:
        return ["lowstate_missing"]
    failures: list[str] = []
    if now - state.received_at > 0.25:
        failures.append("lowstate_stale")
    if not state.standing:
        failures.extend(state.balance_details or ("not_stably_standing",))
    if not state.motion_mode_verified:
        failures.append(f"motion_mode_unverified:{state.motion_mode_name!r}")
    if not state.controller_ownership_verified:
        failures.append("arm_sdk_ownership_unverified")
    if not state.motor_status_verified:
        failures.append("motor_status_unverified")
    if not state.motor_state_healthy:
        failures.extend(state.motor_faults or ("motor_state_unhealthy",))
    if any(not math.isfinite(value) for value in (*state.arm_q, *state.arm_dq)):
        failures.append("arm_state_non_finite")
    if max(abs(value) for value in state.arm_dq) > 0.25:
        failures.append("arm_velocity_too_high")
    return list(dict.fromkeys(failures))


def wait_for_preflight(hardware: UnitreeArmHardware, timeout_s: float) -> RobotState:
    deadline = time.monotonic() + timeout_s
    last_failures: list[str] = ["initializing"]
    while time.monotonic() < deadline:
        state = hardware.latest_state()
        last_failures = state_failures(state, time.monotonic())
        if not last_failures and state is not None:
            return state
        time.sleep(0.05)
    raise RuntimeError("preflight failed: " + ", ".join(last_failures))


def command(
    q: tuple[float, ...],
    *,
    weight: float,
    mode_machine: int,
    kp: float,
    kd: float,
) -> ArmCommand:
    return ArmCommand(
        q=q,
        dq=(0.0,) * ARM_JOINT_COUNT,
        kp=(kp,) * ARM_JOINT_COUNT,
        kd=(kd,) * ARM_JOINT_COUNT,
        weight=min(max(weight, 0.0), 1.0),
        mode_machine=mode_machine,
        published_at=time.monotonic(),
    )


def interpolate(
    start: tuple[float, ...], target: tuple[float, ...], ratio: float
) -> tuple[float, ...]:
    ratio = smoothstep(ratio)
    return tuple((1.0 - ratio) * a + ratio * b for a, b in zip(start, target, strict=True))


def publish_segment(
    hardware: UnitreeArmHardware,
    *,
    start_q: tuple[float, ...],
    target_q: tuple[float, ...],
    start_weight: float,
    target_weight: float,
    duration_s: float,
    mode_machine: int,
    control_hz: float,
    kp: float,
    kd: float,
    stop: threading.Event,
    baseline: tuple[float, ...],
    observation: Observation,
) -> None:
    started = time.monotonic()
    period = 1.0 / control_hz
    while not stop.is_set():
        elapsed = time.monotonic() - started
        ratio = min(elapsed / duration_s, 1.0)
        q = interpolate(start_q, target_q, ratio)
        weight = (1.0 - smoothstep(ratio)) * start_weight + smoothstep(ratio) * target_weight
        hardware.publish(
            command(q, weight=weight, mode_machine=mode_machine, kp=kp, kd=kd)
        )
        state = hardware.latest_state()
        failures = state_failures(state, time.monotonic())
        # Ownership is established before publishing and our own traffic is
        # fingerprinted by the adapter. Any later failure stops the trajectory.
        if failures:
            raise RuntimeError("runtime gate failed: " + ", ".join(failures))
        assert state is not None
        observation.update(state, baseline)
        selected_error = abs(state.arm_q[RIGHT_SHOULDER_PITCH_OFFSET] - q[RIGHT_SHOULDER_PITCH_OFFSET])
        if selected_error > 0.35:
            raise RuntimeError(f"following error exceeded 0.35 rad: {selected_error:.6f}")
        if ratio >= 1.0:
            return
        time.sleep(max(0.0, period - (time.monotonic() - started - elapsed)))


def hold(
    hardware: UnitreeArmHardware,
    *,
    q: tuple[float, ...],
    seconds: float,
    mode_machine: int,
    control_hz: float,
    kp: float,
    kd: float,
    stop: threading.Event,
    baseline: tuple[float, ...],
    observation: Observation,
) -> None:
    if seconds <= 0.0:
        return
    deadline = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < deadline:
        hardware.publish(command(q, weight=1.0, mode_machine=mode_machine, kp=kp, kd=kd))
        state = hardware.latest_state()
        failures = state_failures(state, time.monotonic())
        if failures:
            raise RuntimeError("hold gate failed: " + ", ".join(failures))
        assert state is not None
        observation.update(state, baseline)
        time.sleep(1.0 / control_hz)


def report_state(state: RobotState) -> dict[str, object]:
    return {
        "motion_mode": state.motion_mode_name,
        "motion_mode_verified": state.motion_mode_verified,
        "mode_machine": state.mode_machine,
        "standing": state.standing,
        "ownership_verified": state.controller_ownership_verified,
        "motor_status_verified": state.motor_status_verified,
        "motor_state_healthy": state.motor_state_healthy,
        "right_shoulder_pitch_rad": state.arm_q[RIGHT_SHOULDER_PITCH_OFFSET],
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    if args.mode == "smoke":
        print(json.dumps({
            "ok": True,
            "mode": "smoke",
            "movement": False,
            "interface": args.interface,
            "domain_id": args.domain_id,
            "expected_motion_mode": args.expected_motion_mode,
            "joint": "right_shoulder_pitch_joint",
            "delta_rad": args.delta,
        }, indent=2))
        return 0

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    hardware = UnitreeArmHardware(
        interface=args.interface,
        domain_id=args.domain_id,
        expected_motion_mode=args.expected_motion_mode,
    )
    hardware.start()
    try:
        initial = wait_for_preflight(hardware, args.preflight_timeout)
        baseline = initial.arm_q
        target = list(baseline)
        target[RIGHT_SHOULDER_PITCH_OFFSET] += args.delta
        lower, upper = DEFAULT_RIGHT_JOINT_LIMITS[0]
        if not lower + 0.05 <= target[RIGHT_SHOULDER_PITCH_OFFSET] <= upper - 0.05:
            raise RuntimeError("right shoulder target violates its joint limit margin")
        target_q = tuple(target)
        preflight = {"ok": True, "mode": "preflight", **report_state(initial)}
        if args.mode == "preflight":
            print(json.dumps(preflight, indent=2))
            return 0

        observation = Observation()
        completed = False
        try:
            publish_segment(
                hardware,
                start_q=baseline,
                target_q=baseline,
                start_weight=0.0,
                target_weight=1.0,
                duration_s=args.weight_ramp_seconds,
                mode_machine=initial.mode_machine,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                stop=stop,
                baseline=baseline,
                observation=observation,
            )
            publish_segment(
                hardware,
                start_q=baseline,
                target_q=target_q,
                start_weight=1.0,
                target_weight=1.0,
                duration_s=args.ramp_seconds,
                mode_machine=initial.mode_machine,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                stop=stop,
                baseline=baseline,
                observation=observation,
            )
            hold(
                hardware,
                q=target_q,
                seconds=args.hold_seconds,
                mode_machine=initial.mode_machine,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                stop=stop,
                baseline=baseline,
                observation=observation,
            )
            completed = not stop.is_set()
        finally:
            # Best effort return and weight release; failures still propagate.
            current = hardware.latest_state()
            start_return = baseline if current is None else current.arm_q
            return_stop = threading.Event()
            publish_segment(
                hardware,
                start_q=start_return,
                target_q=baseline,
                start_weight=1.0,
                target_weight=1.0,
                duration_s=args.return_seconds,
                mode_machine=initial.mode_machine,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                stop=return_stop,
                baseline=baseline,
                observation=observation,
            )
            publish_segment(
                hardware,
                start_q=baseline,
                target_q=baseline,
                start_weight=1.0,
                target_weight=0.0,
                duration_s=args.weight_ramp_seconds,
                mode_machine=initial.mode_machine,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                stop=return_stop,
                baseline=baseline,
                observation=observation,
            )

        time.sleep(0.15)
        final = hardware.latest_state()
        if final is None:
            raise RuntimeError("no final lowstate received")
        final_error = abs(final.arm_q[RIGHT_SHOULDER_PITCH_OFFSET] - baseline[RIGHT_SHOULDER_PITCH_OFFSET])
        passed = bool(
            completed
            and observation.maximum_selected_displacement_rad >= 0.025
            and observation.maximum_other_drift_rad <= 0.02
            and final_error <= 0.03
        )
        result = {
            "ok": passed,
            "mode": "execute",
            "joint": "right_shoulder_pitch_joint",
            "requested_delta_rad": args.delta,
            "baseline_rad": baseline[RIGHT_SHOULDER_PITCH_OFFSET],
            "target_rad": target_q[RIGHT_SHOULDER_PITCH_OFFSET],
            "maximum_measured_displacement_rad": observation.maximum_selected_displacement_rad,
            "maximum_other_arm_drift_rad": observation.maximum_other_drift_rad,
            "final_return_error_rad": final_error,
            "final": report_state(final),
        }
        print(json.dumps(result, indent=2))
        return 0 if passed else 1
    finally:
        hardware.close()


if __name__ == "__main__":
    raise SystemExit(main())
