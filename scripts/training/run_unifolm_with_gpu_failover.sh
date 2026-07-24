#!/usr/bin/env bash
set -euo pipefail

workspace="${G1_VLA_WORKSPACE:-/home/aarav/Documents/g1-bunny-vla-workspace}"
launcher="${workspace}/scripts/training/run_unifolm_v29_67real.sh"
primary_gpus="${TRAINING_GPUS:-0,1}"
fallback_gpus="${FALLBACK_TRAINING_GPUS:-1}"
run_id="${RUN_ID:?RUN_ID is required}"
max_steps="${MAX_TRAIN_STEPS:-4000}"
run_root="${workspace}/runs/unifolm_plush_touch"
status_dir="${workspace}/runs/automation/gpu-failover"
status="${status_dir}/${run_id}.json"

mkdir -p "${status_dir}"

write_status() {
  local phase="$1" active_gpus="$2" detail="${3:-}"
  /home/aarav/miniconda3/envs/g1-unifolm-train/bin/python -c \
    'import json,os,sys,time; from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
q=p.with_suffix(".json.partial")
q.write_text(json.dumps({"schema_version":1,"phase":sys.argv[2],"active_gpus":sys.argv[3],"detail":sys.argv[4],"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")
os.replace(q,p)' \
    "${status}" "${phase}" "${active_gpus}" "${detail}"
}

checkpoint_step() {
  basename "$1" | sed -E 's/^steps_([0-9]+)_action_model\.pt$/\1/'
}

write_status training "${primary_gpus}" "primary distributed attempt"
set +e
TRAINING_GPUS="${primary_gpus}" "${launcher}" full
primary_rc=$?
set -e
if (( primary_rc == 0 )); then
  write_status complete "${primary_gpus}" "primary attempt completed"
  exit 0
fi

if [[ "${primary_gpus}" == "${fallback_gpus}" ]]; then
  write_status failed "${primary_gpus}" "training exited ${primary_rc}; no distinct fallback"
  exit "${primary_rc}"
fi

# A distributed process cannot remove a failed rank in place. Confirm that the
# fallback GPU is independently healthy, then warm-start a new single-GPU run
# from the newest portable action checkpoint. Never consume a potentially
# partial DeepSpeed state after a PCIe failure.
IFS=',' read -r -a fallback_ids <<< "${fallback_gpus}"
for gpu in "${fallback_ids[@]}"; do
  nvidia-smi --id="${gpu}" --query-gpu=pci.bus_id --format=csv,noheader >/dev/null
done

failed_run="${run_root}/${run_id}"
latest_checkpoint="$(
  find "${failed_run}/checkpoints" -maxdepth 1 -type f \
    -name 'steps_*_action_model.pt' -size +1M -print 2>/dev/null \
    | sort -V | tail -1
)"
completed_steps=0
if [[ -n "${latest_checkpoint}" ]]; then
  completed_steps="$(checkpoint_step "${latest_checkpoint}")"
fi
remaining_steps="$((max_steps - completed_steps))"
(( remaining_steps > 0 )) || {
  write_status failed "${fallback_gpus}" "primary failed after final portable checkpoint"
  exit "${primary_rc}"
}

fallback_run_id="${run_id}-gpu1-recovery"
fallback_parent="${latest_checkpoint:-${INITIAL_ACTION_CHECKPOINT:-}}"
if [[ -n "${fallback_parent}" ]]; then
  [[ -s "${fallback_parent}" ]] || {
    write_status failed "${fallback_gpus}" "fallback checkpoint is missing"
    exit "${primary_rc}"
  }
else
  # No portable checkpoint exists yet: safely restart from the configured
  # official base model rather than consuming distributed partial state.
  remaining_steps="${max_steps}"
fi
write_status recovering "${fallback_gpus}" \
  "primary exit=${primary_rc}; checkpoint=${fallback_parent:-official-base}; remaining=${remaining_steps}"

RUN_ID="${fallback_run_id}" \
MAX_TRAIN_STEPS="${remaining_steps}" \
TRAINING_GPUS="${fallback_gpus}" \
INITIAL_ACTION_CHECKPOINT="${fallback_parent}" \
RESUME_FROM_CHECKPOINT="" \
  "${launcher}" full
write_status complete "${fallback_gpus}" \
  "fallback completed from ${fallback_parent}"
printf '%s\n' "${fallback_run_id}" > "${status_dir}/${run_id}.effective_run_id"
