#!/usr/bin/env bash
set -euo pipefail

root="${1:-$HOME/Documents/g1-bunny-vla-workspace}"
session="${2:-g1-vla-overnight}"
conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$root/logs"
tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"

tmux new-session -d -s "$session" -n asset-regression \
  "source '$conda_sh'; conda activate unitree_sim_env_isaac50; export G1_BUNNY_PROJECT_ROOT='$root'; while true; do date -Is; python -u '$root/scripts/create_bunny_proxy_usd.py' --output '$root/assets/g1_bunny_proxy.usda' --headless --device cuda:0; sleep 1800; done 2>&1 | tee -a '$root/logs/overnight_asset.log'"
tmux new-window -t "$session" -n hdf5-validation \
  "source '$conda_sh'; conda activate unitree_sim_env_isaac50; export PYTHONPATH='$root/src'; while true; do date -Is; python -m g1_bunny_vla.validate_dataset '$root/data/hdf5'; sleep 600; done 2>&1 | tee -a '$root/logs/overnight_hdf5.log'"
tmux new-window -t "$session" -n rlds-validation \
  "source '$conda_sh'; conda activate g1-bunny-data; while true; do date -Is; python '$root/scripts/check_rlds.py' '$root/data/rlds/g1_bunny_stop/1.0.0'; sleep 600; done 2>&1 | tee -a '$root/logs/overnight_rlds.log'"
tmux new-window -t "$session" -n status \
  "while true; do clear; date -Is; du -sh '$root/data/hdf5' '$root/data/rlds' '$root/assets'; tail -n 5 '$root/logs/overnight_asset.log'; sleep 30; done"
tmux list-windows -t "$session"
