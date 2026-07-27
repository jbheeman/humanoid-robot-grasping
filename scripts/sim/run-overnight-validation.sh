#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s REPLAY_JSON OUTPUT_DIRECTORY\n' "$0" >&2
  exit 2
fi

REPLAY_JSON="$(realpath "$1")"
OUTPUT_DIRECTORY="$(realpath -m "$2")"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
UNITREE_ROOT="${UNITREE_SIM_ROOT:-${ISAAC_VALIDATION_ROOT:-$PWD/.isaac-validation}/unitree_sim_isaaclab}"
ISAACLAB_RUNNER="${ISAACLAB_ROOT:?set ISAACLAB_ROOT to the Isaac Lab installation}/isaaclab.sh"

if [[ ! -x "$ISAACLAB_RUNNER" ]]; then
  printf 'Isaac Lab runner is missing: %s\n' "$ISAACLAB_RUNNER" >&2
  exit 3
fi
mkdir -p "$OUTPUT_DIRECTORY"

run_case() {
  local name="$1"
  local velocity="$2"
  local acceleration="$3"
  local jerk="$4"
  local deadman="$5"
  local actuator_profile="$6"
  "$ISAACLAB_RUNNER" -p "$PROJECT_ROOT/scripts/sim/isaac-g1-replay.py" \
    "$REPLAY_JSON" \
    --output "$OUTPUT_DIRECTORY/$name.json" \
    --project-root "$PROJECT_ROOT" \
    --unitree-sim-root "$UNITREE_ROOT" \
    --headless \
    --device cuda:0 \
    --max-velocity "$velocity" \
    --max-acceleration "$acceleration" \
    --max-jerk "$jerk" \
    --deadman-s "$deadman" \
    --actuator-profile "$actuator_profile"
}

run_case baseline_official 1.0 4.0 30.0 0.75 unitree_official
run_case baseline_sdk_numeric 1.0 4.0 30.0 0.75 sdk_numeric
run_case faster_sdk_numeric 1.5 6.0 40.0 0.75 sdk_numeric
run_case extended_deadman_experiment 1.5 6.0 40.0 2.0 sdk_numeric

python3 - "$OUTPUT_DIRECTORY" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
cases = []
failures = []
for path in sorted(directory.glob("*.json")):
    if path.name == "summary.json":
        continue
    value = json.loads(path.read_text())
    name = path.stem
    if value.get("dds_enabled") is not False or value.get("ros_enabled") is not False:
        failures.append(f"{name}: robot transport was enabled")
    if value.get("transport_import_guard_passed") is not True:
        failures.append(f"{name}: transport import guard did not pass")
    if len((value.get("joint_mapping") or {}).get("body") or {}) != 29:
        failures.append(f"{name}: complete 29-DOF mapping was not verified")
    error = value.get("joint_tracking_error_rad_p95")
    if error is None or error > 0.25:
        failures.append(f"{name}: joint tracking error p95 {error!r} exceeds 0.25 rad")
    unexpected_rejections = {
        code: count
        for code, count in (value.get("target_rejections") or {}).items()
        if code not in {"stale_target"}
    }
    if unexpected_rejections:
        failures.append(f"{name}: unexpected target rejections {unexpected_rejections}")
    cases.append(
        {
            "case": path.stem,
            "proximity_duration_s": value.get("proximity_duration_s"),
            "minimum_hand_to_bunny_m": value.get("minimum_hand_to_bunny_m"),
            "bunny_displacement_m": value.get("bunny_displacement_m"),
            "joint_tracking_error_rad_p95": value.get("joint_tracking_error_rad_p95"),
            "controller_final_state": (value.get("controller_final") or {}).get("state"),
            "realtime_factor": value.get("realtime_factor"),
            "target_rejections": value.get("target_rejections"),
            "actuator_profile": value.get("actuator_profile"),
        }
    )
if len(cases) != 4:
    failures.append(f"expected 4 replay cases, found {len(cases)}")
summary = {
    "schema_version": 1,
    "scope": "downstream_joint_replay_only",
    "passed": not failures,
    "failures": failures,
    "cases": cases,
}
(directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if not failures else 1)
PY
