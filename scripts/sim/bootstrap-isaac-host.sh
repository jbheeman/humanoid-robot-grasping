#!/usr/bin/env bash
set -euo pipefail

UNITREE_SIM_COMMIT="e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc"
SIM_ROOT="${ISAAC_VALIDATION_ROOT:-$PWD/.isaac-validation}"
UNITREE_ROOT="$SIM_ROOT/unitree_sim_isaaclab"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$GPU_NAME" == *"RTX 50"* ]]; then
  ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.0}"
else
  ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1}"
fi

mkdir -p "$SIM_ROOT"
if [[ ! -d "$UNITREE_ROOT/.git" ]]; then
  git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git "$UNITREE_ROOT"
fi
git -C "$UNITREE_ROOT" fetch origin "$UNITREE_SIM_COMMIT"
git -C "$UNITREE_ROOT" checkout --detach "$UNITREE_SIM_COMMIT"
git -C "$UNITREE_ROOT" lfs pull

if [[ ! -d "$UNITREE_ROOT/assets" ]]; then
  (
    cd "$UNITREE_ROOT"
    bash fetch_assets.sh
  )
fi

if [[ -n "${ISAACLAB_ROOT:-}" && -x "$ISAACLAB_ROOT/isaaclab.sh" ]]; then
  printf 'Using existing Isaac Lab: %s\n' "$ISAACLAB_ROOT"
elif [[ -x "$SIM_ROOT/IsaacLab/isaaclab.sh" ]]; then
  printf 'Using existing Isaac Lab: %s\n' "$SIM_ROOT/IsaacLab"
else
  printf '%s\n' \
    "Isaac Lab is not installed." \
    "Run Unitree's pinned installer interactively once:" \
    "  cd $UNITREE_ROOT && bash auto_setup_env.sh $ISAAC_SIM_VERSION unitree_sim_env" \
    "Then export ISAACLAB_ROOT to that installation and rerun this script."
  exit 3
fi

printf 'unitree_sim_root=%s\n' "$UNITREE_ROOT"
printf 'unitree_sim_commit=%s\n' "$UNITREE_SIM_COMMIT"
printf 'gpu=%s\n' "$GPU_NAME"
printf 'isaac_sim_version=%s\n' "$ISAAC_SIM_VERSION"
