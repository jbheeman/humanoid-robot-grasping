#!/usr/bin/env bash
set -euo pipefail

run_id=${1:-moving_block_review_v7}
root=/home/aarav/Documents/g1-bunny-vla-workspace
mkdir -p "$root/logs" "$root/datasets/$run_id"
for suffix in gpu0 gpu1 finalize; do
  tmux has-session -t "${run_id}-${suffix}" 2>/dev/null && {
    echo "session already exists: ${run_id}-${suffix}" >&2; exit 2;
  }
done
tmux new-session -d -s "${run_id}-gpu0" "$root/scripts/run_moving_block_review_suite.sh 0 $run_id"
tmux new-session -d -s "${run_id}-gpu1" "$root/scripts/run_moving_block_review_suite.sh 1 $run_id"
tmux new-session -d -s "${run_id}-finalize" "$root/scripts/finalize_moving_block_review.sh $run_id"
echo "started $run_id: 10 head + 1 angled-left + 1 angled-right cases"

