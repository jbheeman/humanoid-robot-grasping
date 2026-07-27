#!/usr/bin/env bash
set -euo pipefail

# Detached, low-memory synthetic pretraining job for the compact trajectory GRU.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NAME="${TRAJECTORY_TRAIN_NAME:-trajectory_gru_synth_pretrain_17h}"
HOURS="${TRAJECTORY_TRAIN_HOURS:-17}"
LOG_DIR="${ROOT_DIR}/runs/training_logs"
OUTPUT="${ROOT_DIR}/models/plushie_detector/${NAME}"
MIN_MEM_AVAILABLE_KIB="${TRAJECTORY_MIN_MEM_AVAILABLE_KIB:-33554432}" # 32 GiB
MAX_SWAP_USED_KIB="${TRAJECTORY_MAX_SWAP_USED_KIB:-1048576}" # 1 GiB
CPUSET="${TRAJECTORY_CPUSET:-0-17}"
BATCH="${TRAJECTORY_BATCH:-131072}"
HIDDEN_SIZE="${TRAJECTORY_HIDDEN_SIZE:-128}"
EVAL_EVERY="${TRAJECTORY_EVAL_EVERY:-1000}"
VALIDATION_BATCHES="${TRAJECTORY_VALIDATION_BATCHES:-8}"
PREFETCH_WORKERS="${TRAJECTORY_PREFETCH_WORKERS:-18}"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

monitor() {
  local pid="$1" resource_log="$2"
  echo "timestamp,mem_available_kib,swap_used_kib,gpu_used_mib,gpu_util_percent" > "${resource_log}"
  while kill -0 "${pid}" 2>/dev/null; do
    local available swap_total swap_free swap_used gpu_line gpu_used gpu_util
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_total="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
    swap_used=$((swap_total - swap_free))
    gpu_line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    gpu_used="${gpu_line%%,*}"; gpu_util="${gpu_line##*,}"
    gpu_used="${gpu_used//[!0-9]/}"; gpu_util="${gpu_util//[!0-9]/}"
    echo "$(date --iso-8601=seconds),${available},${swap_used},${gpu_used:-0},${gpu_util:-0}" >> "${resource_log}"
    if (( available < MIN_MEM_AVAILABLE_KIB || swap_used > MAX_SWAP_USED_KIB )); then
      echo "[$(date --iso-8601=seconds)] SAFETY STOP: available=${available}KiB swap=${swap_used}KiB" | tee -a "${LOG_DIR}/${NAME}.log"
      # The trainer is a single Python process.  Avoid signaling a shared
      # shell process group, which could include the monitor itself.
      kill -TERM "${pid}" 2>/dev/null || true
      return
    fi
    sleep 15
  done
}

echo "[$(date --iso-8601=seconds)] starting ${NAME} for ${HOURS}h" | tee "${LOG_DIR}/${NAME}.log"
taskset -c "${CPUSET}" nice -n 10 ionice -c 2 -n 7 \
  "${ROOT_DIR}/.venv/bin/python" scripts/training/train_trajectory_forecaster.py \
    --synthetic-pretrain --hours "${HOURS}" --batch "${BATCH}" --hidden-size "${HIDDEN_SIZE}" \
    --eval-every "${EVAL_EVERY}" --validation-batches "${VALIDATION_BATCHES}" \
    --prefetch-workers "${PREFETCH_WORKERS}" --output "${OUTPUT}" \
    >> "${LOG_DIR}/${NAME}.log" 2>&1 &
MAIN_PID=$!
monitor "${MAIN_PID}" "${LOG_DIR}/${NAME}.resources.csv" &
MONITOR_PID=$!
if wait "${MAIN_PID}"; then
  STATUS=0
else
  STATUS=$?
fi
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
exit "${STATUS}"
