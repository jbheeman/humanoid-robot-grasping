#!/usr/bin/env bash
set -uo pipefail

# Run three bounded offline pilots in one tmux-visible workflow. No ROS or
# physical robot transport is imported by the training entry point.

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
PILOT_DIR="${PILOT_DIR:-${ROOT_DIR}/configs/vla/dynamic_pilots}"
OUTPUT_DIR="${ROOT_DIR}/runs/unifolm_plush_touch"
LOG_DIR="${ROOT_DIR}/logs/unifolm_dynamic_pilots"
STATUS_PATH="${OUTPUT_DIR}/dynamic-pilots.status.json"

mkdir -p "${LOG_DIR}"
rm -f "${STATUS_PATH}"

export G1_PLUSH_SHARED_STATS="${ROOT_DIR}/datasets/plush_touch_canonical_block_v28/SHARED_TRAIN_STATS_75_REAL.json"
export TFDS_DATA_DIR="${ROOT_DIR}/datasets/plush_touch_rlds_block_v28"
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${TRAIN_ENV}/bin:${PATH}"

run_variant() {
    local name="$1"
    local gpu="$2"
    local config="${PILOT_DIR}/${name}.yaml"
    local run_id="dynamic-pilot-${name}"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local log="${LOG_DIR}/${name}.log"
    local rendezvous_port="$((29600 + gpu))"

    if [[ ! -s "${config}" ]]; then
        echo "missing pilot config: ${config}" >&2
        return 10
    fi
    if [[ -e "${run_dir}" ]]; then
        echo "refusing to overwrite ${run_dir}" >&2
        return 11
    fi
    local free_gb
    free_gb="$(df -BG --output=avail "${ROOT_DIR}" | tail -1 | tr -dc '0-9')"
    if (( free_gb < 100 )); then
        echo "disk guard: only ${free_gb}GB free before ${name}" >&2
        return 12
    fi

    (
        cd "${ROOT_DIR}/unifolm-vla" || exit 20
        CUDA_VISIBLE_DEVICES="${gpu}" "${TRAIN_ENV}/bin/accelerate" launch \
            --config_file "${ROOT_DIR}/configs/vla/accelerate_plush_touch.yaml" \
            --num_processes 1 \
            --main_process_port "${rendezvous_port}" \
            src/unifolm_vla/training/train_unifolm_vla.py \
            --config_yaml "${config}" \
            --run_id "${run_id}"
    ) > "${log}" 2>&1
}

started_at="$(date --iso-8601=seconds)"

# Equal-budget single-frame control and right-hand loss ablation run together.
run_variant t1-all23 0 &
pid_control=$!
run_variant t1-right9 1 &
pid_right=$!

control_rc=0
right_rc=0
wait "${pid_control}" || control_rc=$?
wait "${pid_right}" || right_rc=$?

# Reuse GPU 0 for the temporal treatment after its equal-budget control.
temporal_rc=0
if [[ "${control_rc}" -eq 0 && "${right_rc}" -eq 0 ]]; then
    run_variant t5s3-right9 0 || temporal_rc=$?
else
    temporal_rc=99
fi

finished_at="$(date --iso-8601=seconds)"
temporary="${STATUS_PATH}.partial"
printf '{"schema_version":1,"started_at":"%s","finished_at":"%s","complete":true,"physical_robot_authorized":false,"exit_codes":{"t1-all23":%s,"t1-right9":%s,"t5s3-right9":%s}}\n' \
    "${started_at}" \
    "${finished_at}" \
    "${control_rc}" \
    "${right_rc}" \
    "${temporal_rc}" > "${temporary}"
mv "${temporary}" "${STATUS_PATH}"

if [[ "${control_rc}" -ne 0 || "${right_rc}" -ne 0 || "${temporal_rc}" -ne 0 ]]; then
    exit 2
fi
