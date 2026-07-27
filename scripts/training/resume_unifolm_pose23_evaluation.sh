#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
LOG="${ROOT_DIR}/logs/unifolm_pose23/evaluate-resume.log"

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "CUDA/NVML is unavailable; reboot the trainer before resuming." >&2
    exit 3
fi

cd "${ROOT_DIR}"
exec "${TRAIN_ENV}/bin/python" \
    scripts/evaluate_unifolm_pose23_pilots.py \
    --root "${ROOT_DIR}" \
    --samples-per-source 96 \
    --gpus 1 0 \
    --reuse-existing >"${LOG}" 2>&1
