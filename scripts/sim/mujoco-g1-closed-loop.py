#!/usr/bin/env python3
"""Transport-free moving-bunny closed loop using MuJoCo arm dynamics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any

import numpy as np

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeError,
    ArmBridgeConfig,
    ArmBridgeController,
    ArmState,
)
from object_tracking.arm_tracking.gravity import UrdfGravityCompensator
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.sim_closed_loop import (
    ObjectObservation,
    SimCommand,
    SimState,
)
from object_tracking.arm_tracking.sim_ipc import LatestPlannerProcess


def percentile(values: list[float], value: float) -> float | None:
    return None if not values else float(np.percentile(values, value))


def load_harness(root: Path) -> dict[str, Any]:
    return runpy.run_path(str(root / "scripts/sim/mujoco-g1-gravity-preflight.py"))


def make_state(
    hardware: Any,
    support: Any,
    *,
    sequence: int,
    simulation_time_s: float,
    speed_m_s: float,
    bunny_x_m: float,
    bunny_start_y_m: float,
    bunny_z_m: float,
    dropout: bool,
) -> SimState:
    observation = None
    if not dropout:
        observation = ObjectObservation(
            track_id=1,
            class_name="bunny",
            confidence=0.90,
            position_m=(
                bunny_x_m,
                bunny_start_y_m - speed_m_s * simulation_time_s,
                bunny_z_m,
            ),
            velocity_m_s=(0.0, -speed_m_s, 0.0),
            observation_time_s=simulation_time_s,
            consecutive_observations=max(1, sequence),
            residual_m=0.0,
        )
    measured = hardware.latest_state()
    return SimState(
        episode_id="mujoco-moving-bunny",
        sequence=sequence,
        simulation_time_s=simulation_time_s,
        calibration_id="lab-sim",
        joint_contract_id=joint_contract_id(),
        body_q_rad=measured.body_q,
        body_dq_rad_s=measured.body_dq,
        support_region=support,
        object_observation=observation,
    )


def wait_for_first_command(
    client: LatestPlannerProcess,
    sequence: int,
    timeout_s: float,
) -> SimCommand:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        command = client.latest_command()
        if command is not None and command.state_sequence == sequence:
            return command
        error = client.metrics().last_error
        if error is not None:
            raise RuntimeError(error)
        time.sleep(0.005)
    raise TimeoutError(f"planner did not answer initial state {sequence}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    parser.add_argument("--planner-hz", type=float, default=30.0)
    parser.add_argument("--bunny-speed", type=float, default=0.05)
    parser.add_argument("--bunny-x", type=float, default=0.36)
    parser.add_argument("--bunny-start-y", type=float, default=0.30)
    parser.add_argument("--bunny-z", type=float, default=0.15)
    parser.add_argument("--dropout-start", type=float, default=2.0)
    parser.add_argument("--dropout-duration", type=float, default=0.15)
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--max-acceleration", type=float, default=4.0)
    parser.add_argument("--max-jerk", type=float, default=30.0)
    parser.add_argument(
        "--diagnostic-deadman",
        type=float,
        default=0.75,
        help="local-only diagnostic override; production remains 0.75 s",
    )
    parser.add_argument("--initial-planner-timeout", type=float, default=90.0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.duration <= 0.0
        or args.physics_hz < 250.0
        or args.physics_hz % 250.0 != 0.0
        or not 0.0 < args.planner_hz <= 60.0
        or not 0.025 <= args.bunny_speed <= 0.10
        or not 0.30 <= args.bunny_x <= 0.45
        or not 0.05 <= args.bunny_start_y <= 0.40
        or not 0.12 <= args.bunny_z <= 0.18
        or args.diagnostic_deadman < 0.75
    ):
        parser.error("invalid duration, rate, or bunny speed")

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Run with: uv run --with mujoco python "
            "scripts/sim/mujoco-g1-closed-loop.py"
        ) from exc

    root = Path(__file__).resolve().parents[2]
    harness = load_harness(root)
    urdf = default_urdf_path(root)
    physics_dt = 1.0 / args.physics_hz
    control_dt = 1.0 / 250.0
    physics_stride = int(round(args.physics_hz / 250.0))
    model, data = harness["load_model"](mujoco, urdf, physics_dt)
    initial_q = harness["body_state"]()
    harness["set_joint_positions"](mujoco, model, data, initial_q)
    support = harness["captured_support"]()
    clock = harness["FakeClock"]()
    hardware = harness["MujocoArmHardware"](
        mujoco,
        model,
        data,
        clock,
        initial_q,
    )
    solver = G1RightArmIK(urdf)
    client = LatestPlannerProcess(
        (
            sys.executable,
            "-u",
            str(root / "scripts/sim/g1-closed-loop-planner.py"),
            "--project-root",
            str(root),
            "--intercept-config",
            str(root / "tests/fixtures/lab-intercept-sim.yaml"),
            "--max-velocity",
            str(args.max_velocity),
            "--max-acceleration",
            str(args.max_acceleration),
            "--max-jerk",
            str(args.max_jerk),
        ),
        cwd=root,
    )
    client.start()
    sequence = 0
    prewarm_time_s = 0.0
    prewarm_commands: list[SimCommand] = []
    first: SimCommand | None = None
    for _ in range(8):
        sequence += 1
        client.submit(
            make_state(
                hardware,
                support,
                sequence=sequence,
                simulation_time_s=prewarm_time_s,
                speed_m_s=args.bunny_speed,
                bunny_x_m=args.bunny_x,
                bunny_start_y_m=args.bunny_start_y,
                bunny_z_m=args.bunny_z,
                dropout=False,
            )
        )
        response = wait_for_first_command(
            client,
            sequence,
            args.initial_planner_timeout,
        )
        prewarm_commands.append(response)
        if response.status == "target":
            first = response
            break
        prewarm_time_s += 1.0 / args.planner_hz
    if first is None:
        client.close()
        raise RuntimeError(
            "planner readiness did not produce a target: "
            f"{[(item.status, item.reason) for item in prewarm_commands]}"
        )

    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            allow_movement=True,
            calibration_id="lab-sim",
            control_hz=250.0,
            target_ttl_s=0.5,
            deadman_s=args.diagnostic_deadman,
            stable_standing_s=0.01,
            startup_settle_s=0.0,
            startup_settle_timeout_s=1.0,
            weight_ramp_s=0.10,
            max_velocity_rad_s=args.max_velocity,
            max_acceleration_rad_s2=args.max_acceleration,
            max_jerk_rad_s3=args.max_jerk,
            max_following_error_rad=0.35,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
        gravity_compensator=UrdfGravityCompensator(urdf),
    )
    controller.enable(session_id="mujoco-closed-loop", calibration_id="lab-sim")

    while controller.state is not ArmState.ARMED:
        controller.tick(clock.monotonic)
        hardware.apply_torques()
        for _ in range(physics_stride):
            mujoco.mj_step(model, data)
            clock.advance(physics_dt)
        if clock.monotonic > 11.0:
            raise RuntimeError(f"controller did not arm: {controller.state_report()}")

    assert first.right_arm_q_rad is not None
    controller.set_target(
        session_id="mujoco-closed-loop",
        sequence=1,
        calibration_id="lab-sim",
        right_arm_q=first.right_arm_q_rad,
        right_arm_tau_ff=first.right_arm_tau_ff_nm,
        pipeline_age_ms=0,
    )
    target_sequence = 1
    reasons = Counter(item.reason for item in prewarm_commands)
    statuses = Counter(item.status for item in prewarm_commands)
    target_rejections: Counter[str] = Counter()
    planner_latencies_ms: list[float] = []
    command_ages_ms = [0.0]
    tracking_errors: list[float] = []
    palm_distances: list[float] = []
    palm_clearances: list[float] = []
    right_side_samples: list[bool] = []
    measured_speeds: list[float] = []
    target_arrival_times = [prewarm_time_s]
    applied_state_sequence = first.state_sequence
    submitted_at = prewarm_time_s
    simulation_time_s = prewarm_time_s
    maximum_progress_m = 0.0
    retraction_events = 0
    retraction_active = False
    measured_q_trace: list[tuple[float, ...]] = [
        hardware.latest_state().arm_q[7:]
    ]
    commanded_q_trace: list[tuple[float, ...]] = []
    initial_palm = solver.forward_kinematics(
        hardware.latest_state().arm_q[7:]
    )[:3, 3]
    wall_started = time.perf_counter()
    deadline = wall_started
    try:
        total_ticks = int(round(args.duration / control_dt))
        for _ in range(total_ticks):
            live_time_s = simulation_time_s - prewarm_time_s
            if simulation_time_s - submitted_at >= 1.0 / args.planner_hz:
                sequence += 1
                dropout = (
                    args.dropout_start
                    <= live_time_s
                    < args.dropout_start + args.dropout_duration
                )
                client.submit(
                    make_state(
                        hardware,
                        support,
                        sequence=sequence,
                        simulation_time_s=simulation_time_s,
                        speed_m_s=args.bunny_speed,
                        bunny_x_m=args.bunny_x,
                        bunny_start_y_m=args.bunny_start_y,
                        bunny_z_m=args.bunny_z,
                        dropout=dropout,
                    )
                )
                submitted_at = simulation_time_s

            latest = client.latest_command()
            if latest is not None and latest.state_sequence > applied_state_sequence:
                applied_state_sequence = latest.state_sequence
                statuses[latest.status] += 1
                reasons[latest.reason] += 1
                planner_latencies_ms.append(latest.planning_latency_ms)
                command_age_s = max(
                    0.0,
                    simulation_time_s - latest.simulation_time_s,
                )
                command_ages_ms.append(command_age_s * 1000.0)
                if latest.status == "target":
                    assert latest.right_arm_q_rad is not None
                    target_sequence += 1
                    try:
                        controller.set_target(
                            session_id="mujoco-closed-loop",
                            sequence=target_sequence,
                            calibration_id="lab-sim",
                            right_arm_q=latest.right_arm_q_rad,
                            right_arm_tau_ff=latest.right_arm_tau_ff_nm,
                            pipeline_age_ms=int(round(command_age_s * 1000.0)),
                        )
                    except ArmBridgeError as exc:
                        target_rejections[exc.code] += 1
                    else:
                        target_arrival_times.append(simulation_time_s)

            controller.tick(clock.monotonic)
            hardware.apply_torques()
            for _ in range(physics_stride):
                mujoco.mj_step(model, data)
                clock.advance(physics_dt)
            simulation_time_s += control_dt

            measured = hardware.latest_state()
            measured_q_trace.append(measured.arm_q[7:])
            palm = solver.forward_kinematics(measured.arm_q[7:])[:3, 3]
            bunny = np.asarray(
                (
                    args.bunny_x,
                    args.bunny_start_y - args.bunny_speed * simulation_time_s,
                    args.bunny_z,
                )
            )
            palm_distances.append(float(np.linalg.norm(palm - bunny)))
            palm_clearances.append(float(support.plane.signed_distance(palm)))
            right_side_samples.append(bool(palm[1] <= bunny[1] + 0.02))
            measured_speeds.append(max(abs(value) for value in measured.arm_dq[7:]))
            progress = float(np.linalg.norm(palm - initial_palm))
            maximum_progress_m = max(maximum_progress_m, progress)
            is_retracted = maximum_progress_m - progress > 0.03
            if is_retracted and not retraction_active:
                retraction_events += 1
            retraction_active = is_retracted
            if hardware.command is not None:
                commanded_q_trace.append(hardware.command.q[7:])
                tracking_errors.append(
                    max(
                        abs(actual - desired)
                        for actual, desired in zip(
                            measured.arm_q[7:],
                            hardware.command.q[7:],
                        )
                    )
                )
            if controller.state in (ArmState.HOLDING, ArmState.FAULT, ArmState.DISARMED):
                break
            if not args.no_realtime_pacing:
                deadline += control_dt
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        client.close()

    target_gaps = [
        (current - previous) * 1000.0
        for previous, current in zip(target_arrival_times, target_arrival_times[1:])
    ]
    def first_invalid_edge(
        trace: list[tuple[float, ...]],
    ) -> dict[str, object] | None:
        for index, (begin, end) in enumerate(zip(trace, trace[1:]), start=1):
            error = solver.validate_joint_path(
                (begin, end),
                support_plane=support,
                edge_step_rad=0.010,
                semantic_edge_step_rad=0.005,
                require_escape_cleared=False,
            )
            if error is not None:
                return {
                    "edge_index": index,
                    "time_s": index * control_dt,
                    "error": error,
                    "begin_q_rad": begin,
                    "end_q_rad": end,
                }
        return None

    measured_path_failure = first_invalid_edge(measured_q_trace)
    commanded_path_failure = first_invalid_edge(commanded_q_trace)
    metrics = client.metrics()
    report = {
        "schema_version": 1,
        "simulator": f"mujoco-{mujoco.__version__}",
        "scope": "right_arm_rigid_body_closed_loop_without_contact",
        "transport_free": True,
        "duration_requested_s": args.duration,
        "duration_completed_s": simulation_time_s - prewarm_time_s,
        "realtime_pacing": not args.no_realtime_pacing,
        "diagnostic_deadman_s": args.diagnostic_deadman,
        "scenario": {
            "bunny_speed_m_s": args.bunny_speed,
            "bunny_x_m": args.bunny_x,
            "bunny_start_y_m": args.bunny_start_y,
            "bunny_z_m": args.bunny_z,
            "dropout_start_s": args.dropout_start,
            "dropout_duration_s": args.dropout_duration,
        },
        "wall_runtime_s": time.perf_counter() - wall_started,
        "first_command": {
            "status": first.status,
            "reason": first.reason,
            "planning_latency_ms": first.planning_latency_ms,
        },
        "prewarm": {
            "commands": len(prewarm_commands),
            "latency_ms": [item.planning_latency_ms for item in prewarm_commands],
            "statuses": [item.status for item in prewarm_commands],
            "reasons": [item.reason for item in prewarm_commands],
        },
        "planner": {
            "statuses": dict(statuses),
            "reasons": dict(reasons),
            "latency_ms_p50": percentile(planner_latencies_ms, 50),
            "latency_ms_p95": percentile(planner_latencies_ms, 95),
            "latency_ms_p99": percentile(planner_latencies_ms, 99),
            "command_age_ms_p95": percentile(command_ages_ms, 95),
            "command_age_ms_max": max(command_ages_ms, default=None),
            "target_gap_ms_p95": percentile(target_gaps, 95),
            "target_gap_ms_max": max(target_gaps, default=None),
            "states_submitted": metrics.states_submitted,
            "pending_states_replaced": metrics.pending_states_replaced,
            "commands_received": metrics.commands_received,
            "stale_commands": metrics.stale_commands,
            "last_error": metrics.last_error,
        },
        "bridge": {
            "state": controller.state.value,
            "fault_reason": controller.fault_reason,
            "hold_reason": controller.hold_reason,
            "target_rejections": dict(target_rejections),
            "metrics": controller.metrics.as_dict(),
        },
        "motion": {
            "target_commands_applied": len(target_arrival_times),
            "maximum_progress_m": maximum_progress_m,
            "retraction_events": retraction_events,
            "minimum_palm_to_bunny_m": min(palm_distances, default=None),
            "palm_clearance_m_min": min(palm_clearances, default=None),
            "measured_path_failure": measured_path_failure,
            "commanded_path_failure": commanded_path_failure,
            "right_side_fraction": (
                sum(right_side_samples) / len(right_side_samples)
                if right_side_samples
                else None
            ),
            "tracking_error_rad_p95": percentile(tracking_errors, 95),
            "maximum_measured_speed_rad_s": max(measured_speeds, default=None),
        },
    }
    report["passed"] = bool(
        simulation_time_s - prewarm_time_s >= args.duration - control_dt
        and controller.state is ArmState.ARMED
        and args.diagnostic_deadman == 0.75
        and not target_rejections
        and metrics.last_error is None
        and metrics.stale_commands == 0
        and len(target_arrival_times) >= 2
        and max(command_ages_ms, default=float("inf")) <= 250.0
        and percentile(planner_latencies_ms, 99) is not None
        and percentile(planner_latencies_ms, 99) <= 100.0  # type: ignore[operator]
        and max(target_gaps, default=float("inf")) <= 250.0
        and percentile(tracking_errors, 95) is not None
        and percentile(tracking_errors, 95) <= 0.10  # type: ignore[operator]
        and maximum_progress_m >= 0.10
        and min(palm_distances, default=float("inf")) <= 0.09
        and measured_path_failure is None
        and commanded_path_failure is None
        and retraction_events == 0
        and sum(right_side_samples) / max(1, len(right_side_samples)) >= 0.90
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
