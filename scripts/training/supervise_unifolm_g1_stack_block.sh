#!/usr/bin/env bash
# Keep the resumable VLA pipeline alive across transient network or package errors.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/unifolm_vla"
PIPELINE_LOG="${LOG_DIR}/unifolm-g1-stack-block.log"
RETRY_SECONDS="${RETRY_SECONDS:-120}"
mkdir -p "$LOG_DIR"

attempt=1
while true; do
  printf '[%s] supervisor: starting pipeline attempt %d\n' "$(date -Is)" "$attempt" >> "$PIPELINE_LOG"
  set +e
  bash "$PROJECT_ROOT/scripts/training/run_unifolm_g1_stack_block.sh" >> "$PIPELINE_LOG" 2>&1
  status=$?
  set -e
  if (( status == 0 )); then
    printf '[%s] supervisor: pipeline completed successfully\n' "$(date -Is)" >> "$PIPELINE_LOG"
    exit 0
  fi
  printf '[%s] supervisor: pipeline exited %d; retrying in %ss\n' \
    "$(date -Is)" "$status" "$RETRY_SECONDS" >> "$PIPELINE_LOG"
  attempt=$((attempt + 1))
  sleep "$RETRY_SECONDS"
done
