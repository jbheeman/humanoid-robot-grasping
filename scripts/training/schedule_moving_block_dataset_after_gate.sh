#!/usr/bin/env bash
set -euo pipefail

gate_pid=${1:?gate PID required}
gate_marker=${2:?gate marker required}
run_id=${3:?run id required}
total=${4:-480}
workers=${5:-8}
root=/home/aarav/Documents/g1-bunny-vla-workspace
status="$root/logs/${run_id}_scheduler.status.json"
wait_result="$root/logs/${run_id}_gate.complete.json"

[[ "$gate_pid" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$total" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$workers" =~ ^[1-8]$ ]] || exit 2
[[ "$gate_marker" == "$root/logs/"* ]] || {
  echo "gate marker must be under $root/logs" >&2
  exit 2
}

write_status() {
  local complete=$1 state=$2 exit_code=$3
  python3 - "$status" "$run_id" "$total" "$workers" "$complete" "$state" "$exit_code" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "complete": sys.argv[5] == "true",
    "run_id": sys.argv[2],
    "episodes": int(sys.argv[3]),
    "workers": int(sys.argv[4]),
    "state": sys.argv[6],
    "exit_code": int(sys.argv[7]),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

on_error() {
  local exit_code=$?
  trap - ERR
  write_status false failed "$exit_code" || true
  exit "$exit_code"
}
trap on_error ERR

write_status false waiting_for_quality_gate 0
python3 "$root/scripts/training/wait_for_completion.py" \
  --pid "$gate_pid" --marker "$gate_marker" --output "$wait_result"

write_status false starting_dataset 0
G1_CONFIRM_DATASET_RUN="$run_id" \
  "$root/scripts/training/start_moving_block_dataset.sh" "$run_id" "$total" "$workers"
write_status true dataset_started 0
trap - ERR
