#!/usr/bin/env bash
set -euo pipefail

case_index=${1:?case index required}
gpu=${2:?GPU index required}
variant=${3:?camera variant required}
run_id=${4:?run id required}

[[ "$case_index" =~ ^([0-9]|1[01])$ ]] || { echo "case index must be 0..11" >&2; exit 2; }
[[ "$gpu" =~ ^[01]$ ]] || { echo "GPU must be 0 or 1" >&2; exit 2; }
case "$variant" in head|angled-left|angled-right) ;; *) exit 2 ;; esac

root=/home/aarav/Documents/g1-bunny-vla-workspace
minimum_free_gb=${G1_DATASET_MIN_FREE_GB:-100}
case_name=$(printf 'case_%02d_%s' "$case_index" "$variant")
output="datasets/${run_id}/${case_name}"
log="logs/${run_id}_${case_name}.log"
seed=$((950000 + case_index * 100))

source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
cd "$root"
mkdir -p "$output" "$(dirname "$log")"
export G1_BUNNY_PROJECT_ROOT="$root"
export UNITREE_SIM_ROOT="$root/unitree_sim_isaaclab"
export PROJECT_ROOT="$root/unitree_sim_isaaclab"

python -u scripts/isaac_generate_plush_touch.py \
  --output-dir "$output" --episodes 1 --seed-start "$seed" \
  --diagnostic-design-index "$case_index" --max-attempt-factor 8 \
  --keep-rejected 4 --min-free-gb "$minimum_free_gb" --camera-variant "$variant" \
  --bunny-usd assets/g1_bunny_real_scale.usda \
  --table-usd assets/real_demo_table.usda \
  --robot-usd assets/generated/g1_brainco/g1_29dof_brainco_official.usd \
  --robot-urdf assets/generated/g1_brainco/g1_29dof_brainco_official.urdf \
  --headless --device "cuda:$gpu" \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/rtx/post/aa/op=2' \
  >"$log" 2>&1
