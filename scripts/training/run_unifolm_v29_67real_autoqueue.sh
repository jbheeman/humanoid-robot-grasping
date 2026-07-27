#!/usr/bin/env bash
set -euo pipefail

workspace=/home/aarav/Documents/g1-bunny-vla-workspace
python=/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python
base_run_id="${AUTO_RUN_BASE_ID:-v29-67real-auto}"
max_attempts="${MAX_AUTO_ATTEMPTS:-3}"
max_steps="${MAX_TRAIN_STEPS:-4000}"
base_seed="${BASE_TRAINING_SEED:-42}"
minimum_free_gb="${MINIMUM_DATA1_FREE_GB:-500}"
initial_checkpoint="${INITIAL_ACTION_CHECKPOINT:-${workspace}/runs/unifolm_plush_touch/v29-67real-motion-mixed-4k/checkpoints/steps_1500_action_model.pt}"
training_gpus="${TRAINING_GPUS:-0,1}"
automation_dir="${workspace}/runs/automation/${base_run_id}"
log_dir="${workspace}/logs/automation/${base_run_id}"
config="${workspace}/configs/vla/v29_67real_motion_history.yaml"
data="${workspace}/datasets/plush_touch_rlds_future1_v29_67real"
statistics="${workspace}/datasets/plush_touch_canonical_v29_67real/SHARED_TRAIN_STATS_75_REAL_RELATIVE_POSE23_FUTURE1.json"

mkdir -p "${automation_dir}" "${log_dir}"
exec > >(tee -a "${log_dir}/autoqueue.log") 2>&1
current_attempt=0
current_run_id=""

write_status() {
  local phase="$1"
  local attempt="$2"
  local run_id="${3:-}"
  local detail="${4:-}"
  "${python}" - "${automation_dir}/AUTOQUEUE_STATUS.json" \
    "${phase}" "${attempt}" "${run_id}" "${detail}" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "phase": sys.argv[2],
    "attempt": int(sys.argv[3]),
    "run_id": sys.argv[4],
    "detail": sys.argv[5],
    "updated_unix": time.time(),
    "robot_execution_authorized": False,
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

record_unexpected_failure() {
  local status=$?
  trap - ERR
  write_status failed "${current_attempt}" "${current_run_id}" "unexpected exit ${status}"
  exit "${status}"
}
trap record_unexpected_failure ERR

require_ready_host() {
  local free_gb gpu
  local -a requested_gpus
  IFS=',' read -r -a requested_gpus <<< "${training_gpus}"
  (( ${#requested_gpus[@]} >= 1 )) || return 1
  for gpu in "${requested_gpus[@]}"; do
    nvidia-smi --id="${gpu}" --query-gpu=pci.bus_id --format=csv,noheader \
      >/dev/null || {
      echo "hardware gate: requested GPU ${gpu} is unavailable" >&2
      return 1
    }
  done
  free_gb="$(df -BG --output=avail "${workspace}/runs" | tail -1 | tr -dc '0-9')"
  (( free_gb >= minimum_free_gb )) || {
    echo "disk gate: ${free_gb}GB free, need ${minimum_free_gb}GB" >&2
    return 1
  }
}

[[ -e /home/aarav/g1-storage-migrate.complete ]] || {
  write_status blocked 0 "" "storage migration is incomplete"
  echo "storage migration is incomplete" >&2
  exit 10
}
[[ -s "${initial_checkpoint}" ]] || {
  write_status blocked 0 "" "initial checkpoint is missing"
  echo "initial checkpoint is missing: ${initial_checkpoint}" >&2
  exit 11
}
require_ready_host || {
  write_status blocked 0 "" "hardware or disk readiness gate failed"
  exit 12
}

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  run_id="${base_run_id}-attempt${attempt}"
  current_attempt="${attempt}"
  current_run_id="${run_id}"
  seed="$((base_seed + attempt - 1))"
  run_dir="${workspace}/runs/unifolm_plush_touch/${run_id}"
  diagnostic_status="${workspace}/runs/diagnostics/${run_id}/PIPELINE_STATUS.json"

  if [[ ! -d "${run_dir}" ]]; then
    require_ready_host || {
      write_status blocked "${attempt}" "${run_id}" "hardware or disk readiness gate failed"
      exit 12
    }
    write_status training "${attempt}" "${run_id}" "seed=${seed}"
    RUN_ID="${run_id}" \
      MAX_TRAIN_STEPS="${max_steps}" \
      TRAINING_SEED="${seed}" \
      TRAINING_GPUS="${training_gpus}" \
      INITIAL_ACTION_CHECKPOINT="${initial_checkpoint}" \
      ACTION_CHECKPOINT_INTERVAL=1000 \
      FULL_STATE_CHECKPOINT_INTERVAL=1000 \
      FULL_STATE_KEEP_LAST=3 \
      "${workspace}/scripts/training/run_unifolm_v29_67real.sh" full
  fi

  require_ready_host || {
    write_status blocked "${attempt}" "${run_id}" "hardware gate failed before validation"
    exit 12
  }
  write_status validating "${attempt}" "${run_id}" ""
  IFS=',' read -r -a evaluation_gpus <<< "${training_gpus}"
  trap - ERR
  set +e
  "${python}" "${workspace}/scripts/training/evaluate_unifolm_run.py" \
    --root "${workspace}" \
    --run-id "${run_id}" \
    --config "${config}" \
    --data-root "${data}" \
    --statistics "${statistics}" \
    --expected-final-step "${max_steps}" \
    --samples-per-source 96 \
    --real-weight 0.75 \
    --gpus "${evaluation_gpus[@]}"
  evaluation_rc=$?
  set -e
  trap record_unexpected_failure ERR

  [[ -s "${diagnostic_status}" ]] || {
    write_status failed "${attempt}" "${run_id}" "validation produced no status"
    exit 20
  }
  phase="$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "${diagnostic_status}")"
  promotion="$("${python}" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1])).get("offline_promotion_passed", False))))' "${diagnostic_status}")"

  if [[ "${promotion}" -eq 1 ]]; then
    write_status promoted "${attempt}" "${run_id}" "${diagnostic_status}"
    ln -sfn "${diagnostic_status}" "${automation_dir}/PROMOTED_PIPELINE_STATUS.json"
    echo "AUTOQUEUE_PROMOTED run=${run_id} attempt=${attempt}"
    exit 0
  fi
  if [[ "${phase}" == validation_rejected ]]; then
    write_status retrying "${attempt}" "${run_id}" "validation rejected; test remained sealed"
    echo "AUTOQUEUE_RETRY run=${run_id} attempt=${attempt}"
    continue
  fi
  if [[ "${phase}" == test_rejected ]]; then
    write_status test_rejected "${attempt}" "${run_id}" "stopping to prevent test-set steering"
    echo "AUTOQUEUE_TEST_REJECTED run=${run_id}; refusing test-guided retry" >&2
    exit 2
  fi

  write_status failed "${attempt}" "${run_id}" "phase=${phase}, evaluator_rc=${evaluation_rc}"
  exit 21
done

write_status exhausted "${max_attempts}" "" "all validation-only attempts rejected"
echo "AUTOQUEUE_EXHAUSTED attempts=${max_attempts}" >&2
exit 3
