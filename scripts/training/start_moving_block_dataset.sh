#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id required}
total=${2:-24}
workers=${3:-8}
root=/home/aarav/Documents/g1-bunny-vla-workspace

[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid run id" >&2; exit 2; }
[[ "$total" =~ ^[1-9][0-9]*$ && "$workers" =~ ^[1-8]$ ]] || exit 2
(( total % workers == 0 )) || { echo "total must be divisible by workers" >&2; exit 2; }
if [[ ${G1_CONFIRM_DATASET_RUN:-} != "$run_id" ]]; then
  echo "approval gate: export G1_CONFIRM_DATASET_RUN=$run_id" >&2
  exit 3
fi

free_gb=$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')
(( free_gb >= 100 )) || { echo "disk guard: only ${free_gb}GB free" >&2; exit 4; }
per_worker=$((total / workers))
mkdir -p "$root/datasets/$run_id" "$root/logs"
for ((worker=0; worker<workers; worker++)); do
  session="${run_id}-worker$(printf '%02d' "$worker")"
  tmux has-session -t "$session" 2>/dev/null && { echo "already running: $session" >&2; exit 5; }
done

for ((worker=0; worker<workers; worker++)); do
  gpu=$((worker % 2))
  variant=head
  (( worker == workers - 2 )) && variant=angled-left
  (( worker == workers - 1 )) && variant=angled-right
  seed=$((1_200_000 + worker * 100_000))
  session="${run_id}-worker$(printf '%02d' "$worker")"
  tmux new-session -d -s "$session" \
    "$root/scripts/run_moving_block_worker.sh $worker $gpu $per_worker $seed $run_id $variant"
done
tmux new-session -d -s "${run_id}-finalize" \
  "$root/scripts/finalize_moving_block_dataset.sh $run_id $total $workers"
echo "started $total episodes across $workers workers; approval-gated run=$run_id"
