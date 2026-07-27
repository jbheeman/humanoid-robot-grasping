#!/usr/bin/env bash
set -uo pipefail

# Offline, event-driven matched action-representation pilots. This script never
# imports ROS, Unitree transport, or the robot bridge.

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
SMOKE_DIR="${SMOKE_DIR:-${ROOT_DIR}/configs/vla/relative_smoke}"
PILOT_DIR="${PILOT_DIR:-${ROOT_DIR}/configs/vla/relative_pilots}"
OUTPUT_DIR="${ROOT_DIR}/runs/unifolm_plush_touch"
LOG_DIR="${ROOT_DIR}/logs/unifolm_pose23"
STATUS_PATH="${OUTPUT_DIR}/pose23-pipeline.status.json"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export G1_PLUSH_SHARED_STATS="${ROOT_DIR}/datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
export TFDS_DATA_DIR="${ROOT_DIR}/datasets/plush_touch_rlds_block_v28"
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${TRAIN_ENV}/bin:${PATH}"

disk_guard() {
    local free_gb
    free_gb="$(df -BG --output=avail "${ROOT_DIR}" | tail -1 | tr -dc '0-9')"
    if (( free_gb < 100 )); then
        echo "disk guard: ${free_gb}GB free; refusing to start another run" >&2
        return 12
    fi
}

run_variant() {
    local config_dir="$1"
    local variant="$2"
    local run_id="$3"
    local gpu="$4"
    local config="${config_dir}/${variant}.yaml"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local log="${LOG_DIR}/${run_id}.log"
    local port="$((29700 + gpu))"
    disk_guard || return $?
    if [[ ! -s "${config}" ]]; then
        echo "missing config: ${config}" >&2
        return 10
    fi
    if [[ -e "${run_dir}" ]]; then
        echo "refusing to overwrite ${run_dir}" >&2
        return 11
    fi
    (
        cd "${ROOT_DIR}/unifolm-vla" || exit 20
        CUDA_VISIBLE_DEVICES="${gpu}" "${TRAIN_ENV}/bin/accelerate" launch \
            --config_file "${ROOT_DIR}/configs/vla/accelerate_plush_touch.yaml" \
            --num_processes 1 \
            --main_process_port "${port}" \
            src/unifolm_vla/training/train_unifolm_vla.py \
            --config_yaml "${config}" \
            --run_id "${run_id}"
    ) >"${log}" 2>&1
}

run_pair() {
    local config_dir="$1"
    local first_variant="$2"
    local first_run="$3"
    local second_variant="$4"
    local second_run="$5"
    run_variant "${config_dir}" "${first_variant}" "${first_run}" 0 &
    local first_pid=$!
    run_variant "${config_dir}" "${second_variant}" "${second_run}" 1 &
    local second_pid=$!
    local first_rc=0
    local second_rc=0
    wait "${first_pid}" || first_rc=$?
    wait "${second_pid}" || second_rc=$?
    if (( first_rc != 0 || second_rc != 0 )); then
        echo "pair failed: ${first_run}=${first_rc} ${second_run}=${second_rc}" >&2
        return 2
    fi
}

started_at="$(date --iso-8601=seconds)"
pipeline_rc=0

# Twenty-step relative smokes validate end-to-end loader, mask, model, and
# checkpoint serialization on both temporal contracts.
run_pair \
    "${SMOKE_DIR}" \
    relative-t1-right9 pose23-smoke-relative-t1-right9 \
    relative-t5s3-right9 pose23-smoke-relative-t5s3-right9 || pipeline_rc=$?

# Equal-budget matched comparisons use both GPUs in each wave.
if (( pipeline_rc == 0 )); then
    run_pair \
        "${PILOT_DIR}" \
        absolute-t1-right9 pose23-pilot-absolute-t1-right9 \
        relative-t1-right9 pose23-pilot-relative-t1-right9 || pipeline_rc=$?
fi
if (( pipeline_rc == 0 )); then
    run_pair \
        "${PILOT_DIR}" \
        absolute-t5s3-right9 pose23-pilot-absolute-t5s3-right9 \
        relative-t5s3-right9 pose23-pilot-relative-t5s3-right9 || pipeline_rc=$?
fi

finished_at="$(date --iso-8601=seconds)"
temporary="${STATUS_PATH}.partial"
printf '{"schema_version":1,"started_at":"%s","finished_at":"%s","complete":true,"exit_code":%s,"physical_robot_authorized":false}\n' \
    "${started_at}" "${finished_at}" "${pipeline_rc}" >"${temporary}"
mv "${temporary}" "${STATUS_PATH}"
exit "${pipeline_rc}"
