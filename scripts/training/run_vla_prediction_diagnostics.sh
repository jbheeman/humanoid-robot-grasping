#!/usr/bin/env bash
set -uo pipefail

# Offline-only UniFoLM diagnostics. This script never imports ROS or opens a robot
# transport. It is intended to run inside one tmux session on the training host.

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
PYTHON_BIN="${PYTHON_BIN:-/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-48}"
SPLIT="${SPLIT:-val}"
GPU_INDEX="${GPU_INDEX:-0}"

CONFIG="${ROOT_DIR}/configs/vla/plush_touch_block_v28_train.yaml"
DATA_ROOT="${ROOT_DIR}/datasets/plush_touch_rlds_block_v28"
STATS="${ROOT_DIR}/datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
BASE_CHECKPOINT="${ROOT_DIR}/models/pretrained/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt"
SELECTED_CHECKPOINT="${ROOT_DIR}/runs/unifolm_plush_touch/block-v28-real-1k/checkpoints/steps_250_action_model.pt"
OUTPUT_DIR="${ROOT_DIR}/runs/diagnostics"
LOG_PATH="${OUTPUT_DIR}/block_v28_prediction_diagnostics.log"
STATUS_PATH="${OUTPUT_DIR}/block_v28_prediction_diagnostics.status.json"

mkdir -p "${OUTPUT_DIR}"
rm -f "${STATUS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

run_diagnostic() {
    local label="$1"
    local checkpoint="$2"
    local output="${OUTPUT_DIR}/block_v28_${label}_prediction_diagnostic_${SPLIT}.json"

    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" \
        "${ROOT_DIR}/scripts/diagnose_unifolm_predictions.py" \
        --checkpoint "${checkpoint}" \
        --output "${output}" \
        --samples-per-source "${SAMPLES_PER_SOURCE}" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --stats "${STATS}" \
        --split "${SPLIT}"
}

started_at="$(date --iso-8601=seconds)"
base_rc=0
selected_rc=0
run_diagnostic base "${BASE_CHECKPOINT}" || base_rc=$?
run_diagnostic selected "${SELECTED_CHECKPOINT}" || selected_rc=$?
finished_at="$(date --iso-8601=seconds)"

temporary_status="${STATUS_PATH}.partial"
printf '{"schema_version":1,"started_at":"%s","finished_at":"%s","split":"%s","samples_per_source":%s,"base_exit_code":%s,"selected_exit_code":%s,"complete":true}\n' \
    "${started_at}" \
    "${finished_at}" \
    "${SPLIT}" \
    "${SAMPLES_PER_SOURCE}" \
    "${base_rc}" \
    "${selected_rc}" > "${temporary_status}"
mv "${temporary_status}" "${STATUS_PATH}"

if [[ "${base_rc}" -ne 0 || "${selected_rc}" -ne 0 ]]; then
    exit 2
fi
