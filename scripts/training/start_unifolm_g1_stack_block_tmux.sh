#!/usr/bin/env bash
# Launch UniFoLM training detached so it survives an SSH disconnect.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${1:-unifolm-g1-stack-block}"
LOG_DIR="${PROJECT_ROOT}/logs/unifolm_vla"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  printf 'Attach with: tmux attach -t %s\n' "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n pipeline -c "$PROJECT_ROOT" \
  "exec bash scripts/training/supervise_unifolm_g1_stack_block.sh"
tmux new-window -d -t "$SESSION" -n monitor -c "$PROJECT_ROOT" \
  "exec tail -F '$LOG_DIR/${SESSION}.log'"
tmux new-window -d -t "$SESSION" -n gpu -c "$PROJECT_ROOT" \
  "exec watch -n 5 nvidia-smi"

printf 'Started tmux session: %s\n' "$SESSION"
printf 'Attach: tmux attach -t %s\n' "$SESSION"
printf 'Windows: pipeline, monitor, gpu (next/previous: Ctrl-b n / Ctrl-b p)\n'
printf 'Log: %s\n' "$LOG_DIR/${SESSION}.log"
