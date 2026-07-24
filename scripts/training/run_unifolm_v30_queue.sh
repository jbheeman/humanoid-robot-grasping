#!/usr/bin/env bash
set -euo pipefail

workspace="${G1_VLA_WORKSPACE:-/home/aarav/Documents/g1-bunny-vla-workspace}"
python="${G1_VLA_PYTHON:-/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python}"
data="${workspace}/datasets/plush_touch_rlds_future1_v29_67real"
statistics="${workspace}/datasets/plush_touch_canonical_v29_67real/SHARED_TRAIN_STATS_V30_PER_HORIZON.json"
config_dir="${workspace}/configs/vla/v30_active_horizon"
base_id="${V30_RUN_BASE_ID:-v30-active-horizon-20260724}"
automation="${workspace}/runs/automation/${base_id}"
log_dir="${workspace}/logs/automation/${base_id}"
primary_gpus="${TRAINING_GPUS:-0,1}"
fallback_gpus="${FALLBACK_TRAINING_GPUS:-1}"

mkdir -p "${automation}" "${log_dir}"
exec > >(tee -a "${log_dir}/queue.log") 2>&1

write_status() {
  local phase="$1" pilot="${2:-}" run_id="${3:-}" detail="${4:-}"
  "${python}" -c 'import json,os,sys,time; from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix(".json.partial")
q.write_text(json.dumps({"schema_version":1,"phase":sys.argv[2],"pilot":sys.argv[3],"run_id":sys.argv[4],"detail":sys.argv[5],"updated_unix":time.time(),"robot_execution_authorized":False},indent=2,sort_keys=True)+"\n")
os.replace(q,p)' \
    "${automation}/QUEUE_STATUS.json" "${phase}" "${pilot}" "${run_id}" "${detail}"
}

healthy_evaluation_gpus() {
  if nvidia-smi --id=0 --query-gpu=pci.bus_id --format=csv,noheader >/dev/null 2>&1 \
    && nvidia-smi --id=1 --query-gpu=pci.bus_id --format=csv,noheader >/dev/null 2>&1; then
    printf '0 1\n'
  else
    nvidia-smi --id=1 --query-gpu=pci.bus_id --format=csv,noheader >/dev/null
    printf '1\n'
  fi
}

selected_checkpoint() {
  "${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$1"
}

run_pilot() {
  local pilot="$1" steps="$2" lr="$3" warmup="$4" parent="${5:-}"
  local run_id="${base_id}-${pilot}"
  local config="${config_dir}/${pilot}.yaml"
  local pipeline_status effective_file effective_run final_step phase
  write_status training "${pilot}" "${run_id}" "gpus=${primary_gpus}"
  RUN_ID="${run_id}" \
  MAX_TRAIN_STEPS="${steps}" \
  TRAINING_GPUS="${primary_gpus}" \
  FALLBACK_TRAINING_GPUS="${fallback_gpus}" \
  INITIAL_ACTION_CHECKPOINT="${parent}" \
  ACTION_MODEL_LR="${lr}" \
  WARMUP_STEPS="${warmup}" \
  REAL_SAMPLE_WEIGHT=8.0 \
  IMAGE_AUG=true \
  ACTION_CHECKPOINT_INTERVAL=1000 \
  FULL_STATE_CHECKPOINT_INTERVAL=1000 \
  FULL_STATE_KEEP_LAST=2 \
  UNIFOLM_CONFIG="${config}" \
  UNIFOLM_DATA_ROOT="${data}" \
  UNIFOLM_STATS="${statistics}" \
    "${workspace}/scripts/training/run_unifolm_with_gpu_failover.sh"

  effective_file="${workspace}/runs/automation/gpu-failover/${run_id}.effective_run_id"
  effective_run="${run_id}"
  [[ ! -s "${effective_file}" ]] || effective_run="$(<"${effective_file}")"
  final_step="$(
    find "${workspace}/runs/unifolm_plush_touch/${effective_run}/checkpoints" \
      -maxdepth 1 -type f -name 'steps_*_action_model.pt' -print \
      | sort -V | tail -1 | sed -E 's/.*steps_([0-9]+)_action_model\.pt/\1/'
  )"
  read -r -a evaluation_gpus <<< "$(healthy_evaluation_gpus)"
  write_status validating "${pilot}" "${effective_run}" "step=${final_step}"
  set +e
  "${python}" "${workspace}/scripts/training/evaluate_unifolm_run.py" \
    --root "${workspace}" \
    --run-id "${effective_run}" \
    --config "${config}" \
    --data-root "${data}" \
    --statistics "${statistics}" \
    --expected-final-step "${final_step}" \
    --samples-per-source 96 \
    --real-weight 0.75 \
    --gate-scope active \
    --gpus "${evaluation_gpus[@]}"
  evaluation_rc=$?
  set -e
  pipeline_status="${workspace}/runs/diagnostics/${effective_run}/PIPELINE_STATUS.json"
  [[ -s "${pipeline_status}" ]] || {
    write_status failed "${pilot}" "${effective_run}" "missing evaluation status"
    return 30
  }
  phase="$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "${pipeline_status}")"
  printf '%s\n' "$(selected_checkpoint "${pipeline_status}")" \
    > "${automation}/${pilot}.selected_checkpoint"
  write_status "${phase}" "${pilot}" "${effective_run}" "evaluation_rc=${evaluation_rc}"
  [[ "${phase}" != complete && "${phase}" != test_rejected ]] || return 20
}

parent=""
run_pilot mixed_active 6000 3e-5 300 "${parent}" || rc=$?
if [[ "${rc:-0}" -eq 20 ]]; then exit 0; elif [[ "${rc:-0}" -ne 0 ]]; then exit "${rc}"; fi
parent="$(<"${automation}/mixed_active.selected_checkpoint")"

unset rc
run_pilot real_only_finish 3000 1e-5 150 "${parent}" || rc=$?
if [[ "${rc:-0}" -eq 20 ]]; then exit 0; elif [[ "${rc:-0}" -ne 0 ]]; then exit "${rc}"; fi
parent="$(<"${automation}/real_only_finish.selected_checkpoint")"

unset rc
run_pilot mixed_fine_motion 5000 2e-5 250 "${parent}" || rc=$?
if [[ "${rc:-0}" -eq 20 ]]; then exit 0; elif [[ "${rc:-0}" -ne 0 ]]; then exit "${rc}"; fi
write_status exhausted "" "" "three controlled v30 pilots completed without promotion"
exit 2
