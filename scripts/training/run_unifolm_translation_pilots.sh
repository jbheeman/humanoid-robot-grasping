#!/usr/bin/env bash
set -uo pipefail

# Offline-only matched loss-weight pilots on the verified future-1 dataset.

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
CONFIG_DIR="${ROOT_DIR}/configs/vla/translation_pilots"
OUTPUT_DIR="${ROOT_DIR}/runs/unifolm_plush_touch"
LOG_DIR="${ROOT_DIR}/logs/unifolm_pose23"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export G1_PLUSH_SHARED_STATS="${ROOT_DIR}/datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
export TFDS_DATA_DIR="${ROOT_DIR}/datasets/plush_touch_rlds_future1_block_v28"
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/unifolm-vla/src:${PYTHONPATH:-}"

run_variant() {
    local variant="$1"
    local gpu="$2"
    local config="${CONFIG_DIR}/${variant}.yaml"
    local run_id="pose23-translation-future1-${variant}"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local log="${LOG_DIR}/${run_id}.log"
    local free_gb
    free_gb="$(df -BG --output=avail "${ROOT_DIR}" | tail -1 | tr -dc '0-9')"
    if (( free_gb < 100 )); then
        echo "disk guard: ${free_gb}GB free; refusing ${run_id}" >&2
        return 12
    fi
    if [[ ! -s "${config}" || -e "${run_dir}" ]]; then
        echo "missing config or existing output for ${run_id}" >&2
        return 11
    fi
    (
        cd "${ROOT_DIR}/unifolm-vla" || exit 20
        CUDA_VISIBLE_DEVICES="${gpu}" "${TRAIN_ENV}/bin/accelerate" launch \
            --config_file "${ROOT_DIR}/configs/vla/accelerate_plush_touch.yaml" \
            --num_processes 1 \
            --main_process_port "$((29920 + gpu))" \
            src/unifolm_vla/training/train_unifolm_vla.py \
            --config_yaml "${config}" \
            --run_id "${run_id}"
    ) >"${log}" 2>&1
}

run_variant xyz3 0 &
pid0=$!
run_variant xyz3-rot01 1 &
pid1=$!
rc0=0
rc1=0
wait "${pid0}" || rc0=$?
wait "${pid1}" || rc1=$?
if (( rc0 != 0 || rc1 != 0 )); then
    echo "translation pilots failed: xyz3=${rc0} xyz3-rot01=${rc1}" >&2
    exit 2
fi
