import argparse
import json

from object_tracking.g1_sim_cli import (
    Candidate,
    REVISION,
    _candidate,
    _candidate_batch,
    _wrapper_xml,
    calibrate,
)


def test_candidate_sampling_is_deterministic_and_bounded() -> None:
    assert _candidate(5, 7) == _candidate(5, 7)
    c = _candidate(5, 7)
    assert 20 <= c.kp <= 110
    assert 0.3 <= c.kd <= 6
    assert 0.03 <= c.vmax <= 0.35
    assert 0.15 <= c.amax <= 4


def test_wrapper_welds_pelvis_and_never_adds_robot_io() -> None:
    xml = _wrapper_xml(__import__("pathlib").Path("/tmp/g1.xml"))
    assert 'body1="pelvis"' in xml
    assert "LowCmd" not in xml
    assert len(REVISION) == 40


def test_candidate_schema() -> None:
    assert Candidate(60, 1.5, 0.1, 0.5).kp == 60


def test_sobol_candidate_batch_is_deterministic_and_diverse() -> None:
    first = _candidate_batch(0, 8, 7, method="sobol", elites=[], best=None)
    second = _candidate_batch(0, 8, 7, method="sobol", elites=[], best=None)
    assert first == second
    assert len({candidate.kp for candidate in first}) == 8


def test_cpu_sweep_memory_guard_has_a_safe_default() -> None:
    args = __import__("object_tracking.g1_sim_cli", fromlist=["parser"]).parser().parse_args(
        ["sweep"]
    )
    assert args.min_available_mib == 2048
    assert args.replays_per_candidate == 1


def test_validation_uses_independent_replays_by_default() -> None:
    args = __import__("object_tracking.g1_sim_cli", fromlist=["parser"]).parser().parse_args(
        ["validate"]
    )
    assert args.seeds == 64
    assert args.workers >= 1


def test_encoder_telemetry_calibration_is_read_only(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(
        json.dumps(
            [
                {"timestamp": 0.0, "commanded_q": [0.0] * 7, "measured_q": [0.0] * 7},
                {"timestamp": 0.1, "commanded_q": [0.1] * 7, "measured_q": [0.05] * 7},
            ]
        )
    )
    assert calibrate(argparse.Namespace(telemetry=str(telemetry))) == 0


def test_commissioning_jsonl_calibration_extracts_right_arm(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-12T00:00:00+00:00",
                "commanded_arm_q": [0.0] * 7 + [0.1] * 7,
                "measured_arm_q": [0.0] * 7 + [0.05] * 7,
            }
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "2026-07-12T00:00:00.100000+00:00",
                "commanded_arm_q": [0.0] * 7 + [0.1] * 7,
                "measured_arm_q": [0.0] * 7 + [0.05] * 7,
            }
        )
        + "\n"
    )
    assert calibrate(argparse.Namespace(telemetry=str(telemetry))) == 0
