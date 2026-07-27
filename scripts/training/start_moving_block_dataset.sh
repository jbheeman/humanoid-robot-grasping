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
worker_pids=()
tmux has-session -t "$run_id" 2>/dev/null && {
  echo "already running: $run_id" >&2
  exit 5
}

for ((worker=0; worker<workers; worker++)); do
  gpu=$((worker % 2))
  variant=head
  (( worker == workers - 2 )) && variant=angled-left
  (( worker == workers - 1 )) && variant=angled-right
  seed=$((1200000 + worker * 100000))
  window="worker$(printf '%02d' "$worker")"
  command="$root/scripts/run_moving_block_worker.sh $worker $gpu $per_worker $seed $run_id $variant"
  if (( worker == 0 )); then
    tmux new-session -d -s "$run_id" -n "$window" "$command"
  else
    tmux new-window -d -t "$run_id:" -n "$window" "$command"
  fi
  pane_pid=$(tmux display-message -p -t "$run_id:$window.0" '#{pane_pid}')
  [[ "$pane_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "could not resolve PID for $run_id:$window" >&2
    exit 6
  }
  worker_pids+=("$pane_pid")
done
[[ ${#worker_pids[@]} -eq $workers ]] || {
  echo "launcher created ${#worker_pids[@]} of $workers workers" >&2
  exit 7
}
tmux new-window -d -t "$run_id:" -n finalize \
  "$root/scripts/training/finalize_moving_block_dataset.sh $run_id $total $workers ${worker_pids[*]}"
echo "started $total episodes across $workers workers; approval-gated run=$run_id"
echo "worker PIDs: ${worker_pids[*]} (event-driven finalizer; no polling loop)"
echo "tmux: $run_id (worker00..worker$(printf '%02d' "$((workers - 1))"), finalize)"
