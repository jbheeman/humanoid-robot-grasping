#!/usr/bin/env bash
set -euo pipefail

workspace=/home/aarav/Documents/g1-bunny-vla-workspace
train_env=/home/aarav/miniconda3/envs/g1-unifolm-train
mode="${1:-pipeline}"
run_id="${RUN_ID:-v29-67real-motion-mixed-4k}"
smoke_run_id="${SMOKE_RUN_ID:-${run_id}-smoke}"
full_steps="${MAX_TRAIN_STEPS:-4000}"
action_save_interval="${ACTION_CHECKPOINT_INTERVAL:-1000}"
full_state_save_interval="${FULL_STATE_CHECKPOINT_INTERVAL:-1000}"
full_state_keep_last="${FULL_STATE_KEEP_LAST:-3}"
training_seed="${TRAINING_SEED:-42}"
initial_action_checkpoint="${INITIAL_ACTION_CHECKPOINT:-}"
training_gpus="${TRAINING_GPUS:-0,1}"
config="${workspace}/configs/vla/v29_67real_motion_history.yaml"
data="${workspace}/datasets/plush_touch_rlds_future1_v29_67real"
stats="${workspace}/datasets/plush_touch_canonical_v29_67real/SHARED_TRAIN_STATS_75_REAL.json"

if [[ "${mode}" == pipeline ]]; then
  exec "${workspace}/scripts/training/run_unifolm_v29_67real_pipeline.sh"
fi

case "${mode}" in
  smoke)
    run_id="${smoke_run_id}"
    steps=20
    save_interval=20
    ;;
  full)
    steps="${full_steps}"
    save_interval="${action_save_interval}"
    ;;
  *)
    echo "usage: $0 {pipeline|smoke|full}" >&2
    exit 2
    ;;
esac

for required in "${config}" "${stats}"; do
  [[ -s "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 3
  }
done
[[ -d "${data}" ]] || {
  echo "missing TFDS root: ${data}" >&2
  exit 3
}
run_dir="${workspace}/runs/unifolm_plush_touch/${run_id}"
[[ ! -e "${run_dir}" ]] || {
  echo "refusing to overwrite ${run_dir}" >&2
  exit 4
}
free_gb="$(df -BG --output=avail "${workspace}/runs" | tail -1 | tr -dc '0-9')"
(( free_gb >= 100 )) || {
  echo "disk guard: ${free_gb}GB free on the runs filesystem; refusing training" >&2
  exit 5
}

cd "${workspace}/unifolm-vla"
IFS=',' read -r -a gpu_ids <<< "${training_gpus}"
num_processes="${#gpu_ids[@]}"
(( num_processes >= 1 )) || {
  echo "TRAINING_GPUS must contain at least one GPU index" >&2
  exit 7
}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${training_gpus}"
export G1_PLUSH_SHARED_STATS="${stats}"
export TFDS_DATA_DIR="${data}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${workspace}/src:${workspace}/unifolm-vla/src:${PYTHONPATH:-}"
export PATH="${train_env}/bin:${PATH}"

extra_args=()
if [[ -n "${initial_action_checkpoint}" ]]; then
  [[ -s "${initial_action_checkpoint}" ]] || {
    echo "initial action checkpoint is missing: ${initial_action_checkpoint}" >&2
    exit 6
  }
  extra_args+=(
    --trainer.action_checkpoint "${initial_action_checkpoint}"
    --trainer.reload_modules action_model
  )
fi

exec "${train_env}/bin/accelerate" launch \
  --config_file "${workspace}/configs/vla/accelerate_plush_touch.yaml" \
  --num_processes "${num_processes}" \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml "${config}" \
  --trainer.max_train_steps "${steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.full_state_save_interval "${full_state_save_interval}" \
  --trainer.full_state_keep_last "${full_state_keep_last}" \
  --trainer.eval_interval "${save_interval}" \
  --seed "${training_seed}" \
  "${extra_args[@]}" \
  --run_id "${run_id}"
