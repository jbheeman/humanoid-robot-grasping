#!/usr/bin/env bash
set -euo pipefail

# Detached Friday-to-Monday training supervisor. Run under setsid/nohup so SSH
# or Codex disconnects cannot stop it.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

REQUESTED_HOURS="${WEEKEND_TRAINING_HOURS:-48}"
WALL_DEADLINE="${WEEKEND_DEADLINE:-$(date -d 'next monday 08:00' '+%Y-%m-%d %H:%M:%S')}"
WALL_DEADLINE_EPOCH="$(date -d "${WALL_DEADLINE}" +%s)"
MAX_BUDGET_EPOCH="$(( $(date +%s) + REQUESTED_HOURS * 3600 ))"
DEADLINE_EPOCH="${WALL_DEADLINE_EPOCH}"
if (( MAX_BUDGET_EPOCH < DEADLINE_EPOCH )); then DEADLINE_EPOCH="${MAX_BUDGET_EPOCH}"; fi
DEADLINE="$(date -d "@${DEADLINE_EPOCH}" '+%Y-%m-%d %H:%M:%S')"
LOG_DIR="${ROOT_DIR}/runs/training_logs"
RUN_ROOT="${ROOT_DIR}/models/plushie_detector"
DATA_ROOT="${ROOT_DIR}/data/plushie"
mkdir -p "${LOG_DIR}"

seconds_left() { echo $(( DEADLINE_EPOCH - $(date +%s) )); }
hours_left() { awk -v seconds="$(seconds_left)" 'BEGIN { printf "%.2f", seconds / 3600 }'; }
require_time() {
  local minimum="$1"
  if (( $(seconds_left) < minimum )); then
    echo "Not enough time remaining before ${DEADLINE}" >&2
    exit 0
  fi
}

# This is intentionally a full-GPU weekend workload. Preserve three CPU cores
# and low scheduling priority for interactive users and system services.
CPUSET="${WEEKEND_CPUSET:-0-16}"
# GB10 uses unified CPU/GPU memory.  We intentionally target high GPU
# utilization while retaining a large OS/CUDA headroom rather than trying to
# allocate 90 percent of memory and risking a machine-level OOM.
MIN_MEM_AVAILABLE_KIB="${WEEKEND_MIN_MEM_AVAILABLE_KIB:-16777216}" # 16 GiB
SAMPLE_INTERVAL_S="${WEEKEND_RESOURCE_SAMPLE_S:-15}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

pkill -TERM -f 'object_tracking.yolo_stream_server' 2>/dev/null || true
sleep 2
"${ROOT_DIR}/.venv/bin/python" scripts/training/build_weekend_holdout.py --dataset "${DATA_ROOT}" \
  | tee "${LOG_DIR}/weekend_holdout.log"
DATA_YAML="${DATA_ROOT}/weekend_grouped_holdout.yaml"

monitor_resources() {
  local pid="$1" name="$2"
  local log="${LOG_DIR}/${name}.resources.csv"
  echo "timestamp,mem_available_kib,swap_used_kib,gpu_used_mib,gpu_util_percent" > "${log}"
  while kill -0 "${pid}" 2>/dev/null; do
    local available swap_total swap_free swap_used gpu_line gpu_used gpu_util
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_total="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
    swap_used=$((swap_total - swap_free))
    gpu_line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    gpu_used="${gpu_line%%,*}"; gpu_util="${gpu_line##*,}"
    gpu_used="${gpu_used//[!0-9]/}"; gpu_util="${gpu_util//[!0-9]/}"
    echo "$(date --iso-8601=seconds),${available},${swap_used},${gpu_used:-0},${gpu_util:-0}" >> "${log}"
    if (( available < MIN_MEM_AVAILABLE_KIB )); then
      echo "[$(date --iso-8601=seconds)] SAFETY STOP ${name}: MemAvailable=${available} KiB" \
        | tee -a "${LOG_DIR}/weekend_supervisor.log"
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      return
    fi
    sleep "${SAMPLE_INTERVAL_S}"
  done
}

run_train() {
  local name="$1" model="$2" budget_hours="$3" phase="$4"
  shift 4
  # Do not use Ultralytics fractional AutoBatch here: its probing phase tried a
  # 16-image 1280 batch and OOMed before it could select a result on GB10.
  # Batch 10 retains substantial unified-memory headroom at this resolution.
  local batch_size="${WEEKEND_BATCH_SIZE:-10}"
  echo "[$(date --iso-8601=seconds)] ${phase}: ${budget_hours} hours remaining=$(hours_left)" \
    | tee -a "${LOG_DIR}/weekend_supervisor.log"
  # A dedicated session lets the monitor terminate the whole training tree if
  # unified memory approaches the safety floor.  Logs stay available after SSH
  # exits and no RAM cache is used.
  setsid taskset -c "${CPUSET}" nice -n 5 ionice -c 2 -n 4 \
    uv run --no-sync yolo detect train \
      model="${model}" data="${DATA_YAML}" epochs=999 time="${budget_hours}" \
      batch="${batch_size}" imgsz=1280 cache=disk device=0 workers=14 \
      save=True save_period=1 project="${RUN_ROOT}" name="${name}" exist_ok=True \
      "$@" > "${LOG_DIR}/${name}.log" 2>&1 &
  local pid=$!
  monitor_resources "${pid}" "${name}" &
  local monitor_pid=$!
  local status=0
  wait "${pid}" || status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  return "${status}"
}

require_time $((3 * 3600))
PRIMARY_HOURS="$(awk -v left="$(hours_left)" 'BEGIN { printf "%.2f", (left < 36 ? left : 36) }')"
PRIMARY_NAME="yolo26l_guarded_primary_1280"
if ! run_train "${PRIMARY_NAME}" "${ROOT_DIR}/models/pretrained/yolo26l.pt" "${PRIMARY_HOURS}" primary \
  patience=50 optimizer=MuSGD lr0=0.00038 lrf=0.882 momentum=0.948 weight_decay=0.00027 \
  warmup_epochs=1.0 box=9.83 cls=0.65 dfl=0.96 close_mosaic=10 \
  mosaic=0.992 mixup=0.427 copy_paste=0.404 scale=0.95 translate=0.275 fliplr=0.304 \
  degrees=0.0 shear=0.0 hsv_h=0.013 hsv_s=0.353 hsv_v=0.194 amp=True plots=True; then
  echo "[$(date --iso-8601=seconds)] primary exited early; retrying safely at 55% memory target" \
    | tee -a "${LOG_DIR}/weekend_supervisor.log"
  PRIMARY_NAME="yolo26l_guarded_recovery_1280"
  RECOVERY_HOURS="$(awk -v left="$(hours_left)" 'BEGIN { printf "%.2f", (left < 36 ? left : 36) }')"
  WEEKEND_BATCH_SIZE=6 run_train "${PRIMARY_NAME}" "${ROOT_DIR}/models/pretrained/yolo26l.pt" "${RECOVERY_HOURS}" recovery \
    patience=50 optimizer=MuSGD lr0=0.00038 lrf=0.882 momentum=0.948 weight_decay=0.00027 \
    warmup_epochs=1.0 box=9.83 cls=0.65 dfl=0.96 close_mosaic=10 \
    mosaic=0.992 mixup=0.427 copy_paste=0.404 scale=0.95 translate=0.275 fliplr=0.304 \
    degrees=0.0 shear=0.0 hsv_h=0.013 hsv_s=0.353 hsv_v=0.194 amp=True plots=True || true
fi

PRIMARY="${RUN_ROOT}/${PRIMARY_NAME}/weights/best.pt"
POLISH="${RUN_ROOT}/yolo26l_weekend_polish_1280/weights/best.pt"
if [[ -f "${PRIMARY}" ]] && (( $(seconds_left) >= 1 * 3600 )); then
  POLISH_HOURS="$(awk -v left="$(hours_left)" 'BEGIN { printf "%.2f", (left < 12 ? left : 12) }')"
  run_train "yolo26l_weekend_polish_1280" "${PRIMARY}" "${POLISH_HOURS}" polish \
    patience=40 optimizer=MuSGD lr0=0.00008 lrf=0.50 momentum=0.948 weight_decay=0.00027 \
    warmup_epochs=0.25 box=9.83 cls=0.65 dfl=0.96 close_mosaic=0 \
    mosaic=0.20 mixup=0.05 copy_paste=0.10 scale=0.50 translate=0.10 fliplr=0.304 \
    degrees=0.0 shear=0.0 hsv_h=0.013 hsv_s=0.353 hsv_v=0.194 amp=True plots=True
fi

if [[ -f "${PRIMARY}" ]]; then
  taskset -c "${CPUSET}" nice -n 5 ionice -c 2 -n 4 \
    uv run --no-sync python scripts/training/select_weekend_model.py \
      --data "${DATA_YAML}" --primary "${PRIMARY}" --polish "${POLISH}" \
      --output "${RUN_ROOT}/yolo26l_weekend_selected" \
      2>&1 | tee "${LOG_DIR}/weekend_select_export.log"
fi
