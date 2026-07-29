from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def passing_result() -> dict[str, object]:
    return {
        "project_commit": "project-commit",
        "project_tracked_dirty": False,
        "unitree_sim_commit": "unitree-commit",
        "runner_sha256": "runner-hash",
        "planner_intercept_config_sha256": "config-hash",
        "planner_urdf_sha256": "urdf-hash",
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


def provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_commit": "project-commit",
        "project_tracked_dirty": False,
        "unitree_sim_commit": "unitree-commit",
        "manifest_sha256": "manifest-hash",
        "intercept_config_sha256": "config-hash",
        "runner_sha256": "runner-hash",
        "planner_sha256": "planner-hash",
        "planner_urdf_sha256": "urdf-hash",
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


def test_single_result_preflight_fails_fast_on_provenance_or_contact(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"schema_version":1}\\n')
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance()))
    result_path = tmp_path / "result.json"
    value = passing_result()
    value["replay_sha256"] = hashlib.sha256(scenario.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(value))
    command = [
        sys.executable,
        str(root / "scripts/sim/audit-single-intercept-result.py"),
        str(result_path),
        str(provenance_path),
        str(scenario),
    ]

    passed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout + passed.stderr

    value["project_commit"] = "wrong-commit"
    result_path.write_text(json.dumps(value))
    mismatch = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert mismatch.returncode == 1
    assert "project_commit mismatch" in mismatch.stdout

    value = passing_result()
    value["replay_sha256"] = hashlib.sha256(scenario.read_bytes()).hexdigest()
    value["contact"] = {"maximum_force_n": 0.0, "duration_s": 0.0}
    result_path.write_text(json.dumps(value))
    no_contact = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert no_contact.returncode == 1
    assert "no >1 N hand/plush contact" in no_contact.stdout


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
    matrix_provenance = provenance()
    matrix_provenance["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    (tmp_path / "provenance.json").write_text(json.dumps(matrix_provenance))
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

    provenance_mismatch = passing_result()
    provenance_mismatch["project_commit"] = "different-commit"
    mismatch_path = results / "canonical_00__r03.json"
    mismatch_path.write_text(json.dumps(provenance_mismatch))
    mismatched = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert mismatched.returncode == 1
    mismatch_summary = json.loads((tmp_path / "summary.json").read_text())
    mismatch_episode = next(
        episode
        for episode in mismatch_summary["episodes"]
        if episode["case"] == "canonical_00" and episode["repetition"] == 3
    )
    assert any("project_commit mismatch" in item for item in mismatch_episode["failures"])

    failed_value = passing_result()
    failed_value["contact"] = {"maximum_force_n": 0.0, "duration_s": 0.0}
    mismatch_path.write_text(json.dumps(failed_value))
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert failed.returncode == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False
    assert summary["canonical_cases_with_fewer_than_10_consecutive_passes"] == [
        "canonical_00"
    ]
