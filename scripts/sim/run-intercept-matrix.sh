#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -d /data1/aarav ]]; then
  DEFAULT_OUTPUT="/data1/aarav/g1-intercept-validation/$STAMP"
else
  DEFAULT_OUTPUT="$PROJECT_ROOT/runs/sim/intercept-matrix/$STAMP"
fi
OUTPUT_DIRECTORY="${1:-$DEFAULT_OUTPUT}"
SCENARIO_DIRECTORY="$OUTPUT_DIRECTORY/scenarios"
RESULT_DIRECTORY="$OUTPUT_DIRECTORY/results"
UNITREE_ROOT="${UNITREE_SIM_ROOT:-${ISAAC_VALIDATION_ROOT:-$PROJECT_ROOT/.isaac-validation}/unitree_sim_isaaclab}"
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
ISAACLAB_RUNNER="${ISAACLAB_ROOT:+$ISAACLAB_ROOT/isaaclab.sh}"

if [[ -z "$ISAAC_PYTHON" && ! -x "$ISAACLAB_RUNNER" ]]; then
  printf '%s\n' \
    "Set ISAAC_PYTHON to the Isaac-enabled Python executable, or" \
    "set ISAACLAB_ROOT to an Isaac Lab installation containing isaaclab.sh." >&2
  exit 3
fi
if [[ ! -f "$UNITREE_ROOT/robots/unitree.py" ]]; then
  printf 'Unitree Isaac assets are missing under %s\n' "$UNITREE_ROOT" >&2
  exit 3
fi

mkdir -p "$SCENARIO_DIRECTORY" "$RESULT_DIRECTORY"
uv run python "$PROJECT_ROOT/scripts/sim/generate-intercept-scenarios.py" \
  "$SCENARIO_DIRECTORY" >"$OUTPUT_DIRECTORY/manifest-generation.log"
MANIFEST="$SCENARIO_DIRECTORY/manifest.json"
PROVENANCE="$OUTPUT_DIRECTORY/provenance.json"
INTERCEPT_CONFIG="$PROJECT_ROOT/tests/fixtures/lab-intercept-sim.yaml"
PLANNER="$PROJECT_ROOT/scripts/sim/g1-closed-loop-planner.py"
RUNNER="$PROJECT_ROOT/scripts/sim/isaac-g1-replay.py"
PLANNER_URDF="$PROJECT_ROOT/.deps/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
uv run python "$PROJECT_ROOT/scripts/sim/write-validation-provenance.py" \
  --project-root "$PROJECT_ROOT" \
  --unitree-root "$UNITREE_ROOT" \
  --manifest "$MANIFEST" \
  --intercept-config "$INTERCEPT_CONFIG" \
  --runner "$RUNNER" \
  --planner "$PLANNER" \
  --urdf "$PLANNER_URDF" \
  --output "$PROVENANCE" >"$OUTPUT_DIRECTORY/provenance-generation.log"

if [[ -n "$ISAAC_PYTHON" ]]; then
  ISAAC_COMMAND=("$ISAAC_PYTHON")
else
  ISAAC_COMMAND=("$ISAACLAB_RUNNER" -p)
fi

readarray -t EPISODES < <(
  python3 - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for case in manifest["cases"]:
    for repetition in range(int(case["repetitions"])):
        print(f"{case['path']}\t{case['name']}__r{repetition:02d}")
PY
)

completed=0
failed_processes=0
total="${#EPISODES[@]}"
for episode in "${EPISODES[@]}"; do
  IFS=$'\t' read -r scenario_name result_name <<<"$episode"
  result="$RESULT_DIRECTORY/$result_name.json"
  log="$RESULT_DIRECTORY/$result_name.log"
  status=0
  if [[ -s "$result" ]]; then
    completed=$((completed + 1))
    printf '[%d/%d] cached %s\n' "$completed" "$total" "$result_name"
  else
    printf '[%d/%d] running %s\n' "$((completed + 1))" "$total" "$result_name"
    "${ISAAC_COMMAND[@]}" "$PROJECT_ROOT/scripts/sim/isaac-g1-replay.py" \
      "$SCENARIO_DIRECTORY/$scenario_name" \
      --output "$result" \
      --project-root "$PROJECT_ROOT" \
      --unitree-sim-root "$UNITREE_ROOT" \
      --headless \
      --device cuda:0 \
      --physics-hz 250 \
      --max-velocity 1.0 \
      --max-acceleration 4.0 \
      --max-jerk 30.0 \
      --deadman-s 0.75 \
      --actuator-profile unitree_official \
      --command-source closed_loop_ipc \
      --planner-launcher local \
      --planner-project-root "$PROJECT_ROOT" \
      --planner-intercept-config "$INTERCEPT_CONFIG" \
      --planner-state-hz 30 \
      --planner-command-ttl-s 0.10 \
      --planner-max-realtime-factor 1.0 \
      --bunny-motion ballistic \
      >"$log" 2>&1
    status=$?
    completed=$((completed + 1))
  fi
  if [[ $status -ne 0 ]]; then
    failed_processes=$((failed_processes + 1))
    printf '[%d/%d] process failed (%d): %s\n' \
      "$completed" "$total" "$status" "$result_name" >&2
  fi
  if [[ $completed -eq 1 ]]; then
    if [[ $status -ne 0 ]]; then
      printf 'Isaac matrix preflight process failed; aborting remaining episodes.\n' >&2
      exit 4
    fi
    uv run python "$PROJECT_ROOT/scripts/sim/audit-single-intercept-result.py" \
      "$result" \
      "$PROVENANCE" \
      "$SCENARIO_DIRECTORY/$scenario_name" \
      >"$OUTPUT_DIRECTORY/preflight-audit.json"
    preflight_status=$?
    if [[ $preflight_status -ne 0 ]]; then
      printf 'Isaac matrix preflight audit failed; see %s\n' \
        "$OUTPUT_DIRECTORY/preflight-audit.json" >&2
      exit 4
    fi
    printf 'Isaac matrix preflight passed: %s\n' "$result_name"
  fi
done

SUMMARY="$OUTPUT_DIRECTORY/summary.json"
uv run python "$PROJECT_ROOT/scripts/sim/summarize-intercept-matrix.py" \
  "$MANIFEST" "$RESULT_DIRECTORY" "$SUMMARY"
audit_status=$?
printf 'results=%s\nprocess_failures=%d\naudit_status=%d\n' \
  "$OUTPUT_DIRECTORY" "$failed_processes" "$audit_status"
exit "$audit_status"
