from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def passing_result() -> dict[str, object]:
    return {
        "dds_enabled": False,
        "ros_enabled": False,
        "transport_import_guard_passed": True,
        "calibrated_frame": "torso_link",
        "joint_mapping": {"body": {str(index): index for index in range(29)}},
        "bunny_motion": "ballistic",
        "object_proxy": {"shape": "capsule"},
        "physx_contact_measurement_available": True,
        "contact": {"maximum_force_n": 2.0, "duration_s": 0.008},
        "target_rejections": {},
        "controller_state_steps": {"ARMED": 1000},
        "joint_tracking_error_rad_p95": 0.05,
        "closed_loop_ipc": {
            "target_commands": 100,
            "expired_commands": 0,
            "stale_commands": 0,
            "last_error": None,
        },
        "bunny_initial_velocity_local_m_s": [0.0, -0.05, 0.0],
        "bunny_final_velocity_local_m_s": [0.0, -0.01, 0.0],
    }


def test_intercept_matrix_generator_is_complete_and_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/sim/generate-intercept-scenarios.py"),
        str(tmp_path),
        "--random-count",
        "2",
        "--canonical-repetitions",
        "2",
        "--seed",
        "7",
    ]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    first_manifest = json.loads((tmp_path / "manifest.json").read_text())
    second = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    second_manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert first.stdout == second.stdout
    assert first_manifest == second_manifest
    assert first_manifest["canonical_case_count"] == 36
    assert first_manifest["heldout_case_count"] == 2
    assert first_manifest["expected_episode_count"] == 74
    assert len(first_manifest["cases"]) == 38

    canonical = [
        case for case in first_manifest["cases"] if case["kind"] == "canonical"
    ]
    assert {
        (case["bunny_x_m"], case["speed_m_s"])
        for case in canonical
    } == {
        (x, speed)
        for x in (0.33, 0.36, 0.40)
        for speed in (0.04, 0.05, 0.07)
    }
    assert {case["table_shift_x_m"] for case in canonical} == {0.0, -0.04}
    assert {case["occlusion"] for case in canonical} == {False, True}

    sample = json.loads((tmp_path / canonical[0]["path"]).read_text())
    assert sample["calibration_id"] == "lab-sim"
    assert sample["object_proxy"]["shape"] == "capsule"
    assert len(sample["frames"][0]["measured_body_q_rad"]) == 29
    assert all(
        "measured_body_q_rad" not in frame for frame in sample["frames"][1:]
    )


def test_matrix_summary_requires_every_canonical_streak_and_95_percent_heldout(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    results = tmp_path / "results"
    results.mkdir()
    cases = [
        {
            "name": f"canonical_{index:02d}",
            "kind": "canonical",
            "repetitions": 10,
        }
        for index in range(36)
    ] + [
        {
            "name": f"heldout_{index:03d}",
            "kind": "heldout",
            "repetitions": 1,
        }
        for index in range(50)
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_episode_count": 410,
                "cases": cases,
            }
        )
    )
    for case in cases:
        for repetition in range(case["repetitions"]):
            (results / f"{case['name']}__r{repetition:02d}.json").write_text(
                json.dumps(passing_result())
            )

    command = [
        sys.executable,
        str(root / "scripts/sim/summarize-intercept-matrix.py"),
        str(manifest_path),
        str(results),
        str(tmp_path / "summary.json"),
    ]
    passed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert json.loads((tmp_path / "summary.json").read_text())["passed"] is True

    failed_value = passing_result()
    failed_value["contact"] = {"maximum_force_n": 0.0, "duration_s": 0.0}
    (results / "canonical_00__r03.json").write_text(json.dumps(failed_value))
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert failed.returncode == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False
    assert summary["canonical_cases_with_fewer_than_10_consecutive_passes"] == [
        "canonical_00"
    ]
