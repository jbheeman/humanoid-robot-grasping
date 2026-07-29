from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.sim_closed_loop import (
    ObjectObservation,
    SimCommand,
    SimState,
    decode_message,
    encode_message,
)
from object_tracking.arm_tracking.sim_ipc import LatestPlannerProcess
from object_tracking.arm_tracking.trajectory import minimum_ruckig_duration_s


def state(sequence: int) -> SimState:
    support = SupportRegion(
        plane=Plane((0.0, 0.0, 1.0), -0.1),
        origin=np.asarray((0.25, 0.3, 0.1)),
        axis_u=np.asarray((1.0, 0.0, 0.0)),
        axis_v=np.asarray((0.0, 1.0, 0.0)),
        minimum_uv=(0.0, -0.6),
        maximum_uv=(0.4, 0.0),
    )
    return SimState(
        episode_id="episode-a",
        sequence=sequence,
        simulation_time_s=sequence / 30.0,
        calibration_id="cal-1",
        joint_contract_id=joint_contract_id(),
        body_q_rad=(0.0,) * 29,
        body_dq_rad_s=(0.0,) * 29,
        support_region=support,
        object_observation=ObjectObservation(
            track_id=1,
            class_name="bunny",
            confidence=1.0,
            position_m=(0.4, 0.2, 0.16),
            velocity_m_s=(0.0, -0.3, 0.0),
            observation_time_s=sequence / 30.0,
            consecutive_observations=sequence + 1,
            residual_m=0.0,
        ),
    )


def wait_for(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_process_is_nonblocking_and_replaces_pending_state_with_newest() -> None:
    worker = """
import json, sys, time
for line_number, line in enumerate(sys.stdin, 1):
    value = json.loads(line)
    if line_number == 1:
        time.sleep(0.1)
    result = {
        "schema_version": 1,
        "kind": "sim_command",
        "episode_id": value["episode_id"],
        "state_sequence": value["sequence"],
        "simulation_time_s": value["simulation_time_s"],
        "source_observation_time_s": value["object_observation"]["observation_time_s"],
        "status": "preview",
        "reason": "test",
        "right_arm_q_rad": None,
        "right_arm_tau_ff_nm": None,
        "target_palm_position_m": None,
        "predicted_crossing_m": None,
        "crossing_time_from_now_s": None,
        "remaining_ruckig_duration_s": None,
        "arrival_slack_s": None,
        "planning_latency_ms": 0.1,
        "ik_step_type": None,
    }
    print(json.dumps(result), flush=True)
"""
    client = LatestPlannerProcess((sys.executable, "-u", "-c", worker))
    client.start()
    try:
        started = time.perf_counter()
        assert client.submit(state(1)) is None
        assert time.perf_counter() - started < 0.02
        time.sleep(0.02)
        client.submit(state(2))
        client.submit(state(3))
        wait_for(lambda: client.metrics().commands_received == 2)

        assert client.latest_command().state_sequence == 3
        metrics = client.metrics()
        assert metrics.states_submitted == 3
        assert metrics.pending_states_replaced == 1
        assert metrics.process_starts == 1
        assert metrics.stale_commands == 0
        assert metrics.last_error is None
    finally:
        client.close()


def test_submit_and_wait_returns_matching_startup_command() -> None:
    worker = """
import json, sys
for line in sys.stdin:
    value = json.loads(line)
    result = {
        "schema_version": 1,
        "kind": "sim_command",
        "episode_id": value["episode_id"],
        "state_sequence": value["sequence"],
        "simulation_time_s": value["simulation_time_s"],
        "source_observation_time_s": value["object_observation"]["observation_time_s"],
        "status": "preview",
        "reason": "startup-ready",
        "right_arm_q_rad": None,
        "right_arm_tau_ff_nm": None,
        "target_palm_position_m": None,
        "predicted_crossing_m": None,
        "crossing_time_from_now_s": None,
        "remaining_ruckig_duration_s": None,
        "arrival_slack_s": None,
        "planning_latency_ms": 0.1,
        "ik_step_type": None,
    }
    print(json.dumps(result), flush=True)
"""
    client = LatestPlannerProcess((sys.executable, "-u", "-c", worker))
    client.start()
    try:
        response = client.submit_and_wait(state(7), timeout_s=2.0)
        assert response.state_sequence == 7
        assert response.reason == "startup-ready"
        assert client.metrics().commands_received == 1
    finally:
        client.close()


def test_submit_and_wait_times_out_without_response() -> None:
    worker = """
import sys, time
for _line in sys.stdin:
    time.sleep(10)
"""
    client = LatestPlannerProcess((sys.executable, "-u", "-c", worker))
    client.start()
    try:
        with pytest.raises(TimeoutError, match="state 3"):
            client.submit_and_wait(state(3), timeout_s=0.05)
    finally:
        client.close()


def test_submit_and_wait_reports_worker_failure() -> None:
    worker = """
import sys
for _line in sys.stdin:
    print("not-json", flush=True)
"""
    client = LatestPlannerProcess((sys.executable, "-u", "-c", worker))
    client.start()
    try:
        with pytest.raises(RuntimeError, match="planner startup failed"):
            client.submit_and_wait(state(4), timeout_s=2.0)
    finally:
        client.close()


def test_actual_worker_is_jsonl_only_and_cannot_open_robot_networks() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts/sim/g1-closed-loop-planner.py"
    wrapper = """
import importlib.abc, runpy, sys
blocked = (
    "unitree_sdk2py", "cyclonedds", "rclpy",
    "object_tracking.ros2_tracking", "object_tracking.ros2_transport",
    "object_tracking.arm_tracking.arm_unitree",
    "object_tracking.arm_tracking.depth_tcp",
)
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("forbidden simulator import: " + fullname)
        return None
def audit(event, args):
    if event in ("socket.connect", "socket.bind"):
        raise RuntimeError("simulator worker attempted network access: " + event)
sys.meta_path.insert(0, Blocker())
sys.addaudithook(audit)
script, project, config = sys.argv[1:]
sys.argv = [
    script, "--project-root", project, "--intercept-config", config,
]
runpy.run_path(script, run_name="__main__")
"""
    message = state(1)
    message = SimState(
        **{
            **message.__dict__,
            "calibration_id": "REPLACE_WITH_ACTIVE_CALIBRATION_ID",
            "object_observation": None,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            wrapper,
            str(worker),
            str(root),
            str(root / "configs/intercept/demo-lane.example.yaml"),
        ],
        cwd=root,
        input=encode_message(message),
        capture_output=True,
        timeout=15.0,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    response = decode_message(lines[0])
    assert isinstance(response, SimCommand)
    assert response.status == "preview"
    assert response.reason == "current_detection_missing"
    assert b"sim_planner_ready" in completed.stderr


@pytest.mark.parametrize("bunny_x_m", (0.33, 0.36, 0.40))
def test_actual_worker_plans_captured_lab_preview_intercept(bunny_x_m: float) -> None:
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts/sim/g1-closed-loop-planner.py"
    start_q = (
        0.2891673744,
        -0.1298251152,
        0.0039188415,
        0.9780925512,
        -0.1113813892,
        -0.0022170816,
        -0.0082091941,
    )
    body_q = [0.0] * 29
    body_q[22:29] = start_q
    origin = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
    axis_u = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
    axis_v = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))
    normal = -np.cross(axis_u, axis_v)
    support = SupportRegion(
        plane=Plane(normal, -float(normal @ origin)),
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        minimum_uv=(0.0, -0.3545687169),
        maximum_uv=(0.34, 0.3254312831),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated_pixel_near_edge"),),
        lateral_margin_m=0.07,
        source="captured_lab",
    )
    message = SimState(
        episode_id=f"captured-lab-{bunny_x_m:.2f}",
        sequence=1,
        simulation_time_s=0.0,
        calibration_id="lab-sim",
        joint_contract_id=joint_contract_id(),
        body_q_rad=tuple(body_q),
        body_dq_rad_s=(0.0,) * 29,
        support_region=support,
        object_observation=ObjectObservation(
            track_id=1,
            class_name="bunny",
            confidence=0.9,
            position_m=(bunny_x_m, 0.30, 0.15),
            velocity_m_s=(0.0, -0.05, 0.0),
            observation_time_s=0.0,
            consecutive_observations=2,
            residual_m=0.0,
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(worker),
            "--project-root",
            str(root),
            "--intercept-config",
            str(root / "tests/fixtures/lab-intercept-sim.yaml"),
        ],
        cwd=root,
        input=encode_message(message),
        capture_output=True,
        timeout=90.0,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    response = decode_message(completed.stdout.splitlines()[0])
    assert isinstance(response, SimCommand)
    assert response.status == "target"
    assert response.reason.startswith("preview_stage:adaptive_table_approach")
    assert response.right_arm_q_rad is not None
    assert response.remaining_ruckig_duration_s is not None
    assert response.crossing_time_from_now_s is not None
    assert response.remaining_ruckig_duration_s < response.crossing_time_from_now_s


def test_actual_worker_closes_loop_through_captured_approach() -> None:
    root = Path(__file__).resolve().parents[1]
    start_q = (
        0.2891673744,
        -0.1298251152,
        0.0039188415,
        0.9780925512,
        -0.1113813892,
        -0.0022170816,
        -0.0082091941,
    )
    origin = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
    axis_u = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
    axis_v = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))
    normal = -np.cross(axis_u, axis_v)
    support = SupportRegion(
        plane=Plane(normal, -float(normal @ origin)),
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        minimum_uv=(0.0, -0.3545687169),
        maximum_uv=(0.34, 0.3254312831),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated_pixel_near_edge"),),
        lateral_margin_m=0.07,
        source="captured_lab",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(root / "scripts/sim/g1-closed-loop-planner.py"),
            "--project-root",
            str(root),
            "--intercept-config",
            str(root / "tests/fixtures/lab-intercept-sim.yaml"),
        ],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    q = start_q
    sim_time = 0.0
    reasons: list[str] = []
    try:
        for sequence in range(1, 16):
            body_q = [0.0] * 29
            body_q[22:29] = q
            state = SimState(
                episode_id="captured-closed-loop",
                sequence=sequence,
                simulation_time_s=sim_time,
                calibration_id="lab-sim",
                joint_contract_id=joint_contract_id(),
                body_q_rad=tuple(body_q),
                body_dq_rad_s=(0.0,) * 29,
                support_region=support,
                object_observation=ObjectObservation(
                    track_id=1,
                    class_name="bunny",
                    confidence=0.9,
                    position_m=(0.36, 0.30 - 0.05 * sim_time, 0.15),
                    velocity_m_s=(0.0, -0.05, 0.0),
                    observation_time_s=sim_time,
                    consecutive_observations=sequence + 1,
                    residual_m=0.0,
                ),
            )
            process.stdin.write(encode_message(state))
            process.stdin.flush()
            response = decode_message(process.stdout.readline())
            assert isinstance(response, SimCommand)
            assert response.status == "target", response.reason
            assert response.right_arm_q_rad is not None
            reasons.append(response.reason)
            edge_duration = minimum_ruckig_duration_s(
                current_position=q,
                target_position=response.right_arm_q_rad,
                maximum_velocity=1.0,
                maximum_acceleration=4.0,
                maximum_jerk=30.0,
            )
            q = response.right_arm_q_rad
            sim_time += max(edge_duration, 1.0 / 30.0)
            if "intercept_local_translation" in response.reason:
                break
        else:
            pytest.fail(f"approach did not complete: {reasons}")
    finally:
        process.stdin.close()
        process.wait(timeout=5.0)

    assert process.returncode == 0
    assert reasons[0].startswith("preview_stage:adaptive_table_approach")
    assert "intercept_local_translation" in reasons[-1]
    assert sim_time < 8.0
