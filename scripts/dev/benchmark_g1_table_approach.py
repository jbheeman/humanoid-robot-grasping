#!/usr/bin/env python3
"""Benchmark the production G1 table approach without robot transports.

The benchmark uses the pinned G1 URDF, the lab close-table calibration, the
production collision checker, and the same measured-state waypoint scheduler
used by the live runtime.  Ruckig is advanced with an ideal measured servo so
planning cost and trajectory duration remain separate metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path
from object_tracking.arm_tracking.runtime import (
    compress_validated_joint_path,
    select_start_escape_waypoint,
)
from object_tracking.arm_tracking.trajectory import ruckig_position_samples


LAB_START_Q = (
    0.2891673744,
    -0.1298251152,
    0.0039188415,
    0.9780925512,
    -0.1113813892,
    -0.0022170816,
    -0.0082091941,
)
LAB_TABLE_ORIGIN = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
LAB_TABLE_AXIS_U = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
LAB_TABLE_AXIS_V = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))
DEFAULT_TARGETS = (
    (0.46, -0.18, 0.105),
    (0.56, -0.08, 0.105),
    (0.66, 0.02, 0.105),
)


def lab_support_region() -> SupportRegion:
    normal = -np.cross(LAB_TABLE_AXIS_U, LAB_TABLE_AXIS_V)
    return SupportRegion(
        plane=Plane(normal, -float(normal @ LAB_TABLE_ORIGIN)),
        origin=LAB_TABLE_ORIGIN,
        axis_u=LAB_TABLE_AXIS_U,
        axis_v=LAB_TABLE_AXIS_V,
        minimum_uv=(0.0, -0.3545687169),
        maximum_uv=(0.34, 0.3254312831),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated_pixel_near_edge"),),
        lateral_margin_m=0.07,
        source="lab_close_table_2026_07_27",
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, fraction * 100.0))


def simulate_ruckig_path(
    path: tuple[tuple[float, ...], ...],
    *,
    control_hz: float,
    maximum_velocity: float,
    maximum_acceleration: float,
    maximum_jerk: float,
    timeout_s: float = 15.0,
    state_validator: Callable[[Sequence[float]], str | None] | None = None,
    strict_waypoint_stop: bool = False,
    residual_lookahead_rad: float = 0.100,
) -> dict[str, Any]:
    dt = 1.0 / control_hz
    otg = Ruckig(7, dt)
    inp = InputParameter(7)
    out = OutputParameter(7)
    measured = np.asarray(path[0], dtype=float)
    inp.current_position = measured.tolist()
    inp.current_velocity = [0.0] * 7
    inp.current_acceleration = [0.0] * 7
    inp.max_velocity = [maximum_velocity] * 7
    inp.max_acceleration = [maximum_acceleration] * 7
    inp.max_jerk = [maximum_jerk] * 7
    target_index = 1
    last_advance = tuple(float(value) for value in measured)
    previous_acceleration = np.zeros(7)
    velocities: list[float] = []
    accelerations: list[float] = []
    jerks: list[float] = []
    waypoint_switches = 0
    steps = 0
    previous_result = Result.Working
    while steps * dt <= timeout_s:
        previous_index = target_index
        if strict_waypoint_stop:
            if previous_result == Result.Finished:
                target_index += 1
            waypoint = None if target_index >= len(path) else path[target_index]
            error = None
        else:
            waypoint, target_index, error = select_start_escape_waypoint(
                measured,
                path,
                target_index,
                last_advance_q_rad=last_advance,
                residual_lookahead_rad=residual_lookahead_rad,
            )
        if error is not None:
            return {"ok": False, "reason": error}
        if target_index > previous_index:
            last_advance = tuple(float(value) for value in measured)
            waypoint_switches += target_index - previous_index
        if waypoint is None:
            return {
                "ok": True,
                "duration_s": round(steps * dt, 4),
                "steps": steps,
                "waypoint_switches": waypoint_switches,
                "maximum_velocity_rad_s": max(velocities, default=0.0),
                "maximum_acceleration_rad_s2": max(accelerations, default=0.0),
                "maximum_jerk_rad_s3": max(jerks, default=0.0),
                "final_error_rad": float(np.max(np.abs(measured - np.asarray(path[-1])))),
            }
        inp.target_position = list(waypoint)
        inp.target_velocity = [0.0] * 7
        inp.target_acceleration = [0.0] * 7
        result = otg.update(inp, out)
        if result not in (Result.Working, Result.Finished):
            return {"ok": False, "reason": f"ruckig:{result}"}
        measured = np.asarray(out.new_position, dtype=float)
        if state_validator is not None:
            validation_error = state_validator(measured)
            if validation_error is not None:
                return {
                    "ok": False,
                    "reason": f"sampled_trajectory:{validation_error}",
                    "step": steps,
                    "target_index": target_index,
                    "q_rad": [round(float(value), 6) for value in measured],
                }
        velocity = np.asarray(out.new_velocity, dtype=float)
        acceleration = np.asarray(out.new_acceleration, dtype=float)
        velocities.append(float(np.max(np.abs(velocity))))
        accelerations.append(float(np.max(np.abs(acceleration))))
        jerks.append(float(np.max(np.abs((acceleration - previous_acceleration) / dt))))
        previous_acceleration = acceleration
        previous_result = result
        out.pass_to_input(inp)
        steps += 1
    return {"ok": False, "reason": "timeout"}


def benchmark_target(
    solver: G1RightArmIK,
    support: SupportRegion,
    target_xyz: tuple[float, float, float],
    *,
    control_hz: float,
    maximum_velocity: float,
    maximum_acceleration: float,
    maximum_jerk: float,
    compression_span_rad: float,
    compression_skip_knots: int,
    strict_waypoint_stop: bool = True,
    residual_lookahead_rad: float = 0.100,
) -> dict[str, Any]:
    target_transform = solver.forward_kinematics(LAB_START_Q)
    target_transform[:3, 3] = target_xyz
    started = time.perf_counter()
    planned = solver.plan_adaptive_table_approach(
        target_transform,
        LAB_START_Q,
        support_plane=support,
        top_clearance_m=0.11,
        final_validation_edge_step_rad=None,
        desired_palm_normal=(0.0, 1.0, 0.0),
        maximum_palm_normal_error_rad=math.radians(25.0),
    )
    adaptive_reason = planned.reason
    route = "adaptive"
    if not planned.ok or planned.q_path is None:
        planned = solver.plan_guided_clearance(
            LAB_START_Q,
            support_plane=support,
            lift_m=0.12,
            forward_m=0.04,
        )
        route = "guided_fallback"
    planning_ms = (time.perf_counter() - started) * 1000.0
    record: dict[str, Any] = {
        "target_xyz_m": target_xyz,
        "route": route,
        "adaptive_reason": adaptive_reason,
        "planning_ms": planning_ms,
        "planning_ok": planned.ok,
        "planning_reason": planned.reason,
    }
    if not planned.ok or planned.q_path is None:
        return record

    validation_times: list[float] = []

    def edge_is_valid(begin: tuple[float, ...], end: tuple[float, ...]) -> bool:
        edge_started = time.perf_counter()
        error = solver.validate_joint_path(
            (begin, end),
            support_plane=support,
            edge_step_rad=0.020,
            semantic_edge_step_rad=0.010,
            require_escape_cleared=False,
        )
        if error is None:
            samples = ruckig_position_samples(
                current_position=begin,
                target_position=end,
                maximum_velocity=maximum_velocity,
                maximum_acceleration=maximum_acceleration,
                maximum_jerk=maximum_jerk,
                sample_period_s=1.0 / control_hz,
            )
            error = solver.validate_joint_path(
                samples,
                support_plane=support,
                edge_step_rad=0.010,
                semantic_edge_step_rad=0.005,
                require_escape_cleared=False,
            )
        validation_times.append((time.perf_counter() - edge_started) * 1000.0)
        return error is None

    compression_started = time.perf_counter()
    compressed = compress_validated_joint_path(
        planned.q_path,
        edge_is_valid,
        maximum_span_rad=compression_span_rad,
        maximum_skip_knots=compression_skip_knots,
    )
    compression_ms = (time.perf_counter() - compression_started) * 1000.0
    record.update(
        {
            "raw_waypoints": len(planned.q_path),
            "compression_ms": compression_ms,
            "compression_edge_checks": len(validation_times),
            "compression_edge_check_ms_p50": percentile(validation_times, 0.50),
            "compression_edge_check_ms_p95": percentile(validation_times, 0.95),
        }
    )
    if compressed is None:
        record["compression_reason"] = "no_valid_compressed_path"
        return record

    dense_started = time.perf_counter()
    dense_error = solver.validate_joint_path(
        compressed,
        support_plane=support,
        edge_step_rad=0.005,
        semantic_edge_step_rad=0.005,
        require_escape_cleared=False,
    )
    dense_validation_ms = (time.perf_counter() - dense_started) * 1000.0
    previous_ruckig_q = np.asarray(compressed[0], dtype=float)

    def validate_ruckig_sample(q: Sequence[float]) -> str | None:
        nonlocal previous_ruckig_q
        current_q = np.asarray(q, dtype=float)
        error = solver.validate_joint_path(
            (previous_ruckig_q, current_q),
            support_plane=support,
            edge_step_rad=0.005,
            semantic_edge_step_rad=0.0025,
            require_escape_cleared=False,
        )
        previous_ruckig_q = current_q
        return error

    record.update(
        {
            "compressed_waypoints": len(compressed),
            "dense_validation_ms": dense_validation_ms,
            "dense_validation_error": dense_error,
            "ruckig": simulate_ruckig_path(
                compressed,
                control_hz=control_hz,
                maximum_velocity=maximum_velocity,
                maximum_acceleration=maximum_acceleration,
                maximum_jerk=maximum_jerk,
                state_validator=validate_ruckig_sample,
                strict_waypoint_stop=strict_waypoint_stop,
                residual_lookahead_rad=residual_lookahead_rad,
            ),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help="repeat for a target matrix; defaults cover the right-side tabletop",
    )
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--max-acceleration", type=float, default=4.0)
    parser.add_argument("--max-jerk", type=float, default=30.0)
    parser.add_argument("--compression-span-rad", type=float, default=0.70)
    parser.add_argument("--compression-skip-knots", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    limits = (
        args.control_hz,
        args.max_velocity,
        args.max_acceleration,
        args.max_jerk,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in limits):
        parser.error("control and trajectory limits must be finite and positive")
    if not 50.0 <= args.control_hz <= 250.0:
        parser.error("--control-hz must be between 50 and 250")
    if not math.isfinite(args.compression_span_rad) or args.compression_span_rad <= 0.0:
        parser.error("--compression-span-rad must be finite and positive")
    if args.compression_skip_knots < 1:
        parser.error("--compression-skip-knots must be positive")
    targets = tuple(tuple(item) for item in (args.target or DEFAULT_TARGETS))
    if not all(len(item) == 3 and all(math.isfinite(value) for value in item) for item in targets):
        parser.error("targets must contain three finite coordinates")

    solver = G1RightArmIK(default_urdf_path(args.repo_root.resolve()))
    support = lab_support_region()
    cases = [
        benchmark_target(
            solver,
            support,
            target,
            control_hz=args.control_hz,
            maximum_velocity=args.max_velocity,
            maximum_acceleration=args.max_acceleration,
            maximum_jerk=args.max_jerk,
            compression_span_rad=args.compression_span_rad,
            compression_skip_knots=args.compression_skip_knots,
        )
        for target in targets
    ]
    report = {
        "schema_version": 1,
        "robot_execution_authorized": False,
        "transport_modules_used": False,
        "control_hz": args.control_hz,
        "limits": {
            "velocity_rad_s": args.max_velocity,
            "acceleration_rad_s2": args.max_acceleration,
            "jerk_rad_s3": args.max_jerk,
        },
        "compression": {
            "maximum_span_rad": args.compression_span_rad,
            "maximum_skip_knots": args.compression_skip_knots,
        },
        "cases": cases,
        "passed": all(
            case.get("planning_ok")
            and case.get("dense_validation_error") is None
            and (case.get("ruckig") or {}).get("ok")
            for case in cases
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
