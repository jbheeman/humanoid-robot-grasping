#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id required}
total=${2:?total required}
workers=${3:?workers required}
shift 3
root=/home/aarav/Documents/g1-bunny-vla-workspace
log="$root/logs/${run_id}_finalize.log"
status="$root/logs/${run_id}_finalize.status.json"
wait_result="$root/logs/${run_id}_workers.complete.json"
exec >"$log" 2>&1

[[ $# -eq $workers ]] || { echo "expected $workers worker PIDs, got $#" >&2; exit 2; }
pid_args=()
for worker_pid in "$@"; do
  [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] || { echo "invalid worker PID: $worker_pid" >&2; exit 2; }
  pid_args+=(--pid "$worker_pid")
done

write_status() {
  local complete=$1
  local state=$2
  local exit_code=$3
  python3 - "$status" "$run_id" "$total" "$complete" "$state" "$exit_code" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "complete": sys.argv[4] == "true",
    "run_id": sys.argv[2],
    "expected_episodes": int(sys.argv[3]),
    "state": sys.argv[5],
    "exit_code": int(sys.argv[6]),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
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

write_status false waiting_for_workers 0
python3 "$root/scripts/training/wait_for_completion.py" \
  "${pid_args[@]}" --output "$wait_result"

source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
cd "$root"
write_status false auditing 0
python scripts/training/audit_moving_block_dataset.py "datasets/$run_id" \
  --expected-count "$total" --output "artifacts/vla_dataset_review/$run_id/dataset_audit.json"
free_gb=$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')
python - "$status" "$run_id" "$total" "$free_gb" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "complete": True,
    "run_id": sys.argv[2],
    "accepted": int(sys.argv[3]),
    "free_disk_gb": int(sys.argv[4]),
    "training_started": False,
    "next_gate": "human dataset approval",
}, indent=2) + "\n")
os.replace(temporary, path)
PY
trap - ERR
