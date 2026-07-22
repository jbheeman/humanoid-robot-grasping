#!/usr/bin/env bash
set -euo pipefail

worker=${1:?worker index required}
gpu=${2:?GPU index required}
episodes=${3:?episode count required}
seed_start=${4:?seed start required}
run_id=${5:?run id required}
variant=${6:?camera variant required}

[[ "$worker" =~ ^[0-9]+$ ]] || exit 2
[[ "$gpu" =~ ^[01]$ ]] || exit 2
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$variant" in head|angled-left|angled-right) ;; *) exit 2 ;; esac

root=/home/aarav/Documents/g1-bunny-vla-workspace
output="$root/datasets/$run_id/worker$(printf '%02d' "$worker")"
log="$root/logs/${run_id}_worker$(printf '%02d' "$worker").log"
minimum_free_gb=${G1_DATASET_MIN_FREE_GB:-100}

source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
cd "$root"
mkdir -p "$output" "$(dirname "$log")"
export G1_BUNNY_PROJECT_ROOT="$root"
export UNITREE_SIM_ROOT="$root/unitree_sim_isaaclab"
export PROJECT_ROOT="$root/unitree_sim_isaaclab"

python -u scripts/isaac_generate_plush_touch.py \
  --output-dir "$output" --episodes "$episodes" --seed-start "$seed_start" \
  --max-attempt-factor 4 --keep-rejected 4 --min-free-gb "$minimum_free_gb" \
  --camera-variant "$variant" \
  --bunny-usd assets/g1_bunny_real_scale.usda \
  --table-usd assets/real_demo_table.usda \
  --robot-usd assets/generated/g1_brainco/g1_29dof_brainco_official.usd \
  --robot-urdf assets/generated/g1_brainco/g1_29dof_brainco_official.urdf \
  --headless --device "cuda:$gpu" \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/rtx/post/aa/op=2' \
  >"$log" 2>&1
