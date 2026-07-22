#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id required}
total=${2:?total required}
workers=${3:?workers required}
root=/home/aarav/Documents/g1-bunny-vla-workspace
log="$root/logs/${run_id}_finalize.log"
status="$root/logs/${run_id}_finalize.status.json"
exec >"$log" 2>&1

while true; do
  active=0
  for ((worker=0; worker<workers; worker++)); do
    session="${run_id}-worker$(printf '%02d' "$worker")"
    tmux has-session -t "$session" 2>/dev/null && active=1
  done
  (( active == 0 )) && break
  sleep 10
done

source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
cd "$root"
python scripts/audit_moving_block_dataset.py "datasets/$run_id" \
  --expected-count "$total" --output "artifacts/vla_dataset_review/$run_id/dataset_audit.json"
free_gb=$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')
python - "$status" "$run_id" "$total" "$free_gb" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "complete": True,
    "run_id": sys.argv[2],
    "accepted": int(sys.argv[3]),
    "free_disk_gb": int(sys.argv[4]),
    "training_started": False,
    "next_gate": "human dataset approval",
}, indent=2) + "\n")
PY
