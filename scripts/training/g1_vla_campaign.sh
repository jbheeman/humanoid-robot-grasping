#!/usr/bin/env bash
set -euo pipefail

workspace="${G1_VLA_WORKSPACE:-/home/aarav/Documents/g1-bunny-vla-workspace}"
python="${G1_VLA_PYTHON:-/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python}"
campaign="${G1_VLA_CAMPAIGN:-v29-67real-adaptive}"
session="g1-vla-${campaign}"
controller="${workspace}/scripts/training/unifolm_campaign.py"
automation="${workspace}/runs/automation/${campaign}"
log_dir="${workspace}/logs/automation/${campaign}"
initial_checkpoint="${INITIAL_ACTION_CHECKPOINT:-}"
command="${1:-status}"

controller_command() {
  "${python}" "${controller}" --workspace "${workspace}" --campaign "${campaign}" "$@"
}

start_campaign() {
  [[ -n "${initial_checkpoint}" && -s "${initial_checkpoint}" ]] || {
    echo "Set INITIAL_ACTION_CHECKPOINT to an existing action-model checkpoint." >&2
    return 2
  }
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "campaign is already running in tmux session ${session}"
    return 0
  fi
  mkdir -p "${automation}" "${log_dir}"
  controller_command resume
  tmux new-session -d -s "${session}" \
    "exec '${python}' '${controller}' \
      --workspace '${workspace}' --campaign '${campaign}' run \
      --initial-checkpoint '${initial_checkpoint}' \
      --config '${workspace}/configs/vla/v29_67real_motion_history.yaml' \
      --data-root '${workspace}/datasets/plush_touch_rlds_future1_v29_67real' \
      --statistics '${workspace}/datasets/plush_touch_canonical_v29_67real/SHARED_TRAIN_STATS_75_REAL_RELATIVE_POSE23_FUTURE1.json' \
      --training-gpus '${TRAINING_GPUS:-1}' \
      --max-steps '${MAX_TRAIN_STEPS:-4000}' \
      --checkpoint-interval '${ACTION_CHECKPOINT_INTERVAL:-1000}' \
      --samples '${VALIDATION_SAMPLES_PER_SOURCE:-96}' \
      --patience '${CAMPAIGN_PATIENCE:-3}' \
      --max-attempts '${CAMPAIGN_MAX_ATTEMPTS:-24}' \
      >>'${log_dir}/campaign.log' 2>&1"
  echo "started ${session}; log: ${log_dir}/campaign.log"
}

case "${command}" in
  start|resume)
    start_campaign
    ;;
  stop-now|stop-after-attempt|backfill|export|plot)
    controller_command "${command}"
    ;;
  status)
    controller_command status
    tmux has-session -t "${session}" 2>/dev/null \
      && echo "tmux=${session} running" \
      || echo "tmux=${session} stopped"
    ;;
  logs)
    tail -n "${LINES:-80}" "${log_dir}/campaign.log"
    ;;
  *)
    echo "usage: $0 {start|status|logs|stop-now|stop-after-attempt|resume|backfill|export|plot}" >&2
    exit 2
    ;;
esac
