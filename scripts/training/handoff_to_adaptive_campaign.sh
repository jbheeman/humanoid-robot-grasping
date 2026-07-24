#!/usr/bin/env bash
set -euo pipefail

old_pid="${1:?usage: $0 OLD_QUEUE_PID INITIAL_ACTION_CHECKPOINT}"
initial_checkpoint="${2:?usage: $0 OLD_QUEUE_PID INITIAL_ACTION_CHECKPOINT}"
workspace="${G1_VLA_WORKSPACE:-/home/aarav/Documents/g1-bunny-vla-workspace}"

[[ "${old_pid}" =~ ^[0-9]+$ ]] || {
  echo "OLD_QUEUE_PID must be numeric" >&2
  exit 2
}
[[ -s "${initial_checkpoint}" ]] || {
  echo "initial checkpoint is missing: ${initial_checkpoint}" >&2
  exit 3
}

# Linux tail uses pidfd/process-exit notification internally; this avoids a
# polling sleep loop while the legacy queue finishes training and validation.
tail --pid="${old_pid}" -f /dev/null

cd "${workspace}"
scripts/training/g1_vla_campaign.sh backfill
INITIAL_ACTION_CHECKPOINT="${initial_checkpoint}" \
TRAINING_GPUS="${TRAINING_GPUS:-1}" \
CAMPAIGN_PATIENCE="${CAMPAIGN_PATIENCE:-3}" \
CAMPAIGN_MAX_ATTEMPTS="${CAMPAIGN_MAX_ATTEMPTS:-24}" \
  scripts/training/g1_vla_campaign.sh start
