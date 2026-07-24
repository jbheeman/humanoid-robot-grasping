#!/usr/bin/env bash
set -euo pipefail

project_root=${G1_BUNNY_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
asset_root=${G1_BUNNY_ASSET_ROOT:-/home/aarav/Documents/g1-bunny-vla-workspace}
sim_root=${UNITREE_SIM_ROOT:-/home/aarav/Documents/g1-bunny-vla-workspace/unitree_sim_isaaclab}
output_root=${G1_INTERCEPT_OUTPUT_ROOT:-/data1/aarav/data-stores/g1-bunny-vla/intercept/oracle-pilot}
physical_gpu=${G1_INTERCEPT_GPU:-1}
attempts=${G1_INTERCEPT_ATTEMPTS:-30}
seed_start=${G1_INTERCEPT_SEED_START:-2700000}

[[ "$physical_gpu" =~ ^[01]$ ]] || { echo "G1_INTERCEPT_GPU must be 0 or 1" >&2; exit 2; }
[[ "$attempts" =~ ^[1-9][0-9]*$ ]] || { echo "G1_INTERCEPT_ATTEMPTS must be positive" >&2; exit 2; }

source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export G1_BUNNY_PROJECT_ROOT="$project_root"
export UNITREE_SIM_ROOT="$sim_root"
export PROJECT_ROOT="$sim_root"

if ! python -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'; then
  echo "CUDA physical GPU $physical_gpu is unavailable; cold-power-cycle the host before retrying." >&2
  exit 75
fi

mkdir -p "$output_root/episodes" "$output_root/logs"
if [[ -e "$output_root/oracle_rollouts.jsonl" ]] || compgen -G "$output_root/episodes/episode_*.hdf5" >/dev/null; then
  echo "Refusing to overwrite an existing pilot: $output_root" >&2
  echo "Set G1_INTERCEPT_OUTPUT_ROOT to a new directory to rerun." >&2
  exit 73
fi
cd "$project_root"
python -u scripts/isaac_generate_plush_touch.py \
  --output-dir "$output_root/episodes" \
  --rollout-output "$output_root/oracle_rollouts.jsonl" \
  --episodes "$attempts" \
  --seed-start "$seed_start" \
  --keep-rejected "$attempts" \
  --min-free-gb 500 \
  --camera-variant head \
  --bunny-usd "$asset_root/assets/g1_bunny_real_scale.usda" \
  --table-usd "$asset_root/assets/real_demo_table.usda" \
  --robot-usd "$asset_root/assets/generated/g1_brainco/g1_29dof_brainco_official.usd" \
  --robot-urdf "$asset_root/assets/generated/g1_brainco/g1_29dof_brainco_official.urdf" \
  --headless \
  --device cuda:0 \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/rtx/post/aa/op=0 --/rtx/indirectDiffuse/denoiser/temporal/enabled=false --/rtx/directLighting/sampledLighting/samplesPerPixel=8' \
  2>&1 | tee "$output_root/logs/oracle_pilot.log"
