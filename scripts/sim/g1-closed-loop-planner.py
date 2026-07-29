#!/usr/bin/env python3
"""Run one production-class G1 simulator planner over JSONL stdin/stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--intercept-config", type=Path, required=True)
parser.add_argument("--project-root", type=Path, default=Path.cwd())
parser.add_argument("--max-velocity", type=float, default=1.0)
parser.add_argument("--max-acceleration", type=float, default=4.0)
parser.add_argument("--max-jerk", type=float, default=30.0)
args = parser.parse_args()

project_root = args.project_root.resolve()
if not (project_root / "src/object_tracking").is_dir():
    parser.error("--project-root does not contain this project")
sys.path.insert(0, str(project_root / "src"))

from object_tracking.arm_tracking.ik_solver import (  # noqa: E402
    G1RightArmIK,
    default_urdf_path,
)
from object_tracking.arm_tracking.interception import (  # noqa: E402
    load_live_intercept_config,
)
from object_tracking.arm_tracking.sim_closed_loop import (  # noqa: E402
    SimState,
    decode_message,
    encode_message,
)
from object_tracking.arm_tracking.sim_planner import (  # noqa: E402
    ClosedLoopInterceptionPlanner,
    SimPlannerConfig,
)


def assert_isolated() -> None:
    forbidden = (
        "unitree_sdk2py",
        "cyclonedds",
        "rclpy",
        "object_tracking.ros2_tracking",
        "object_tracking.ros2_transport",
        "object_tracking.arm_tracking.arm_unitree",
        "object_tracking.arm_tracking.depth_tcp",
    )
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == token or name.startswith(f"{token}.") for token in forbidden)
    )
    if loaded:
        raise RuntimeError(f"simulator planner loaded forbidden modules: {loaded}")


def log(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), file=sys.stderr, flush=True)


def main() -> int:
    assert_isolated()
    profile = load_live_intercept_config(args.intercept_config)
    planner = ClosedLoopInterceptionPlanner(
        profile,
        G1RightArmIK(default_urdf_path(project_root)),
        SimPlannerConfig(
            maximum_velocity_rad_s=args.max_velocity,
            maximum_acceleration_rad_s2=args.max_acceleration,
            maximum_jerk_rad_s3=args.max_jerk,
        ),
    )
    assert_isolated()
    log(
        "sim_planner_ready",
        intercept_config=str(args.intercept_config.resolve()),
        urdf=str(default_urdf_path(project_root)),
        robot_transport_imports=False,
        started_at=time.time(),
    )
    for payload in sys.stdin.buffer:
        try:
            message = decode_message(payload)
            if not isinstance(message, SimState):
                raise ValueError("planner accepts only sim_state messages")
            response = planner.plan(message)
            assert_isolated()
            sys.stdout.buffer.write(encode_message(response))
            sys.stdout.buffer.flush()
            log(
                "sim_planner_command",
                sequence=response.state_sequence,
                status=response.status,
                reason=response.reason,
                planning_latency_ms=round(response.planning_latency_ms, 3),
                ruckig_duration_s=response.remaining_ruckig_duration_s,
                measured_right_q_rad=[
                    round(value, 5) for value in message.right_arm_q_rad
                ],
                target_right_q_rad=(
                    None
                    if response.right_arm_q_rad is None
                    else [round(value, 5) for value in response.right_arm_q_rad]
                ),
                waypoint_index=planner._path_index,
                waypoint_count=(
                    None if planner._path is None else len(planner._path)
                ),
            )
        except (OSError, ValueError) as exc:
            log("sim_planner_message_rejected", error=f"{type(exc).__name__}: {exc}")
            return 2
    return 0


raise SystemExit(main())
