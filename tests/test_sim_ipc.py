from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import numpy as np

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
