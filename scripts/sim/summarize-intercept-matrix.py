#!/usr/bin/env python3
"""Strictly audit a completed ballistic Isaac interception matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_result(value: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if value.get("dds_enabled") is not False or value.get("ros_enabled") is not False:
        failures.append("robot transport enabled")
    if value.get("transport_import_guard_passed") is not True:
        failures.append("transport import guard failed")
    if value.get("calibrated_frame") != "torso_link":
        failures.append("calibrated frame is not torso_link")
    if len((value.get("joint_mapping") or {}).get("body") or {}) != 29:  # type: ignore[union-attr]
        failures.append("29-DOF joint mapping missing")
    if value.get("bunny_motion") != "ballistic":
        failures.append("bunny was not dynamic")
    if (value.get("object_proxy") or {}).get("shape") != "capsule":  # type: ignore[union-attr]
        failures.append("plush proxy is not a capsule")
    if value.get("physx_contact_measurement_available") is not True:
        failures.append("filtered PhysX contact measurement unavailable")
    contact = value.get("contact") or {}
    if float(contact.get("maximum_force_n") or 0.0) <= 1.0:  # type: ignore[union-attr]
        failures.append("no >1 N hand/plush contact")
    if float(contact.get("duration_s") or 0.0) < 0.004:  # type: ignore[union-attr]
        failures.append("hand/plush contact was shorter than one 250 Hz step")
    if value.get("target_rejections"):
        failures.append(f"target rejections: {value['target_rejections']}")
    if int((value.get("controller_state_steps") or {}).get("FAULT") or 0) > 0:  # type: ignore[union-attr]
        failures.append("bridge entered FAULT")
    tracking_error = value.get("joint_tracking_error_rad_p95")
    if tracking_error is None or float(tracking_error) > 0.25:
        failures.append(f"joint tracking p95 is {tracking_error!r}")

    closed_loop = value.get("closed_loop_ipc") or {}
    if int(closed_loop.get("target_commands") or 0) < 1:  # type: ignore[union-attr]
        failures.append("planner emitted no target")
    if int(closed_loop.get("expired_commands") or 0) != 0:  # type: ignore[union-attr]
        failures.append("planner command expired")
    if int(closed_loop.get("stale_commands") or 0) != 0:  # type: ignore[union-attr]
        failures.append("stale planner command observed")
    if closed_loop.get("last_error") is not None:  # type: ignore[union-attr]
        failures.append(f"planner process error: {closed_loop['last_error']}")  # type: ignore[index]

    initial_velocity = value.get("bunny_initial_velocity_local_m_s")
    final_velocity = value.get("bunny_final_velocity_local_m_s")
    if not (
        isinstance(initial_velocity, list)
        and isinstance(final_velocity, list)
        and len(initial_velocity) == len(final_velocity) == 3
    ):
        failures.append("bunny velocity evidence missing")
    else:
        initial_speed = abs(float(initial_velocity[1]))
        delta = sum(
            (float(final) - float(initial)) ** 2
            for initial, final in zip(initial_velocity, final_velocity)
        ) ** 0.5
        if initial_speed <= 0.0 or delta < max(0.02, 0.25 * initial_speed):
            failures.append(
                f"bunny was not arrested/deflected enough: delta={delta:.4f} m/s"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = args.results_directory.resolve()
    episodes: list[dict[str, object]] = []
    missing: list[str] = []
    canonical_passes: dict[str, list[bool]] = {}
    heldout_passes: list[bool] = []

    for case in manifest["cases"]:
        name = str(case["name"])
        repetitions = int(case["repetitions"])
        case_passes: list[bool] = []
        for repetition in range(repetitions):
            result_name = f"{name}__r{repetition:02d}.json"
            path = results / result_name
            if not path.is_file():
                missing.append(result_name)
                case_passes.append(False)
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            failures = audit_result(value)
            passed = not failures
            case_passes.append(passed)
            episodes.append(
                {
                    "case": name,
                    "kind": case["kind"],
                    "repetition": repetition,
                    "passed": passed,
                    "failures": failures,
                    "result": result_name,
                }
            )
        if case["kind"] == "canonical":
            canonical_passes[name] = case_passes
        else:
            heldout_passes.extend(case_passes)

    expected = int(manifest["expected_episode_count"])
    passed_count = sum(bool(item["passed"]) for item in episodes)
    canonical_failures = [
        name
        for name, passes in canonical_passes.items()
        if len(passes) < 10 or not all(passes[:10])
    ]
    heldout_rate = (
        0.0
        if not heldout_passes
        else sum(heldout_passes) / len(heldout_passes)
    )
    overall_rate = passed_count / expected if expected else 0.0
    passed = (
        not missing
        and len(episodes) == expected
        and not canonical_failures
        and len(heldout_passes) >= 50
        and heldout_rate >= 0.95
        and overall_rate >= 0.95
    )
    summary = {
        "schema_version": 1,
        "passed": passed,
        "expected_episodes": expected,
        "completed_episodes": len(episodes),
        "passed_episodes": passed_count,
        "overall_pass_rate": round(overall_rate, 6),
        "canonical_cases_with_fewer_than_10_consecutive_passes": canonical_failures,
        "heldout_episode_count": len(heldout_passes),
        "heldout_pass_rate": round(heldout_rate, 6),
        "missing_results": missing,
        "episodes": episodes,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
