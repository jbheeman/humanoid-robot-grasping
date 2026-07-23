#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
CONFIG_DIR="${ROOT_DIR}/configs/vla/absolute_future_pilots"
DATA_ROOT_PREFIX="${ROOT_DIR}/datasets/plush_touch_rlds_future"
STATS_ROOT="${ROOT_DIR}/datasets/plush_touch_canonical_block_v28"
OUTPUT_DIR="${ROOT_DIR}/runs/unifolm_plush_touch"
LOG_DIR="${ROOT_DIR}/logs/unifolm_pose23"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/unifolm-vla/src:${PYTHONPATH:-}"

run_variant() {
    local lag="$1"
    local gpu="$2"
    local variant="future${lag}-absolute-t1-right9"
    local run_id="pose23-${variant}"
    local config="${CONFIG_DIR}/${variant}.yaml"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local stats="${STATS_ROOT}/SHARED_TRAIN_STATS_75_REAL_ABSOLUTE_POSE23_FUTURE${lag}.json"
    local data_root="${DATA_ROOT_PREFIX}${lag}_block_v28"
    if [[ ! -s "${config}" || ! -s "${stats}" || -e "${run_dir}" ]]; then
        echo "missing input or existing output for ${run_id}" >&2
        return 11
    fi
    (
        cd "${ROOT_DIR}/unifolm-vla" || exit 20
        export G1_PLUSH_SHARED_STATS="${stats}"
        export TFDS_DATA_DIR="${data_root}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${TRAIN_ENV}/bin/accelerate" launch \
            --config_file "${ROOT_DIR}/configs/vla/accelerate_plush_touch.yaml" \
            --num_processes 1 \
            --main_process_port "$((29940 + gpu))" \
            src/unifolm_vla/training/train_unifolm_vla.py \
            --config_yaml "${config}" \
            --run_id "${run_id}"
    ) >"${LOG_DIR}/${run_id}.log" 2>&1
}

run_variant 1 0 &
pid0=$!
run_variant 3 1 &
pid1=$!
rc0=0
rc1=0
wait "${pid0}" || rc0=$?
wait "${pid1}" || rc1=$?
if (( rc0 != 0 || rc1 != 0 )); then
    echo "absolute future pilots failed: future1=${rc0} future3=${rc1}" >&2
    exit 2
fi
