from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.sim_closed_loop import (
    ObjectObservation,
    SequenceGate,
    SimCommand,
    SimState,
    decode_message,
    encode_message,
)


def support_region() -> SupportRegion:
    return SupportRegion(
        plane=Plane((0.0, 0.0, 1.0), -0.1),
        origin=np.asarray((0.25, 0.3, 0.1)),
        axis_u=np.asarray((1.0, 0.0, 0.0)),
        axis_v=np.asarray((0.0, 1.0, 0.0)),
        minimum_uv=(0.0, -0.6),
        maximum_uv=(0.4, 0.0),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated"),),
        source="test",
    )


def state(sequence: int = 2, time_s: float = 1.25) -> SimState:
    return SimState(
        episode_id="episode-a",
        sequence=sequence,
        simulation_time_s=time_s,
        calibration_id="cal-1",
        joint_contract_id=joint_contract_id(),
        body_q_rad=(0.0,) * 29,
        body_dq_rad_s=(0.0,) * 29,
        support_region=support_region(),
        object_observation=ObjectObservation(
            track_id=1,
            class_name="bunny",
            confidence=1.0,
            position_m=(0.42, 0.15, 0.16),
            velocity_m_s=(0.0, -0.3, 0.0),
            observation_time_s=time_s,
            consecutive_observations=8,
            residual_m=0.004,
        ),
    )


def test_state_round_trip_preserves_measured_state_and_support_region() -> None:
    original = state()
    decoded = decode_message(encode_message(original))

    assert isinstance(decoded, SimState)
    assert decoded.to_dict() == original.to_dict()
    assert decoded.right_arm_q_rad == (0.0,) * 7
    assert decoded.support_region.source == "test"


def test_target_command_round_trip_contains_timing_evidence() -> None:
    original = SimCommand(
        episode_id="episode-a",
        state_sequence=12,
        simulation_time_s=2.0,
        source_observation_time_s=1.98,
        status="target",
        reason="committed",
        right_arm_q_rad=(0.1,) * 7,
        right_arm_tau_ff_nm=(0.3,) * 7,
        target_palm_position_m=(0.4, -0.05, 0.16),
        predicted_crossing_m=(0.4, 0.0, 0.16),
        crossing_time_from_now_s=0.4,
        remaining_ruckig_duration_s=0.31,
        arrival_slack_s=0.09,
        planning_latency_ms=8.2,
        ik_step_type="intercept_local_translation",
    )

    assert decode_message(encode_message(original)) == original


def test_sequence_gate_discards_duplicates_and_time_rewinds() -> None:
    gate = SequenceGate()

    assert gate.accept(state(2, 1.0))
    assert not gate.accept(state(2, 1.1))
    assert not gate.accept(state(3, 0.9))
    assert gate.accept(state(4, 1.2))
    assert (gate.accepted, gate.discarded) == (2, 2)


@pytest.mark.parametrize(
    "change",
    [
        {"body_q_rad": (0.0,) * 28},
        {"body_dq_rad_s": (float("nan"),) + (0.0,) * 28},
        {"simulation_time_s": -1.0},
        {"episode_id": ""},
        {"calibration_id": ""},
    ],
)
def test_invalid_state_is_rejected(change: dict[str, object]) -> None:
    values = state().to_dict()
    values.update(change)
    with pytest.raises(ValueError):
        SimState.from_dict(values)


def test_non_target_command_cannot_smuggle_joint_motion() -> None:
    with pytest.raises(ValueError, match="non-target"):
        SimCommand(
            episode_id="episode-a",
            state_sequence=1,
            simulation_time_s=0.1,
            source_observation_time_s=None,
            status="hold",
            reason="acquiring",
            right_arm_q_rad=(0.0,) * 7,
        )


def test_contract_import_does_not_load_robot_transport_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    source = """
import json, sys
import object_tracking.arm_tracking.sim_closed_loop
forbidden = ("unitree_sdk2py", "cyclonedds", "rclpy")
loaded = sorted(
    name for name in sys.modules
    if any(name == token or name.startswith(token + ".") for token in forbidden)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
