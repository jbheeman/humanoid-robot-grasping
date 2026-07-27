#!/usr/bin/env bash
set -euo pipefail

# Full-quality YOLO26L training for GB10.  `epochs` is only a ceiling: Ultralytics
# stops after `patience` validation epochs without an mAP improvement.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NAME="${TRAIN_NAME:-yolo26l_full_1280}"
LOG_DIR="${ROOT_DIR}/runs/training_logs"
RUN_ROOT="${ROOT_DIR}/models/plushie_detector"
MIN_MEM_AVAILABLE_KIB="${TRAIN_MIN_MEM_AVAILABLE_KIB:-20971520}" # 20 GiB
mkdir -p "${LOG_DIR}"

"${ROOT_DIR}/.venv/bin/python" scripts/training/build_weekend_holdout.py \
  --dataset "${ROOT_DIR}/data/plushie" > "${LOG_DIR}/${NAME}.holdout.log"

monitor() {
  while kill -0 "${MAIN_PID}" 2>/dev/null; do
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    echo "$(date --iso-8601=seconds),${available}" >> "${LOG_DIR}/${NAME}.resources.csv"
    if (( available < MIN_MEM_AVAILABLE_KIB )); then
      echo "SAFETY STOP: only ${available} KiB MemAvailable" >&2
      kill -TERM -- "-${MAIN_PID}" 2>/dev/null || true
      return
    fi
    sleep 15
  done
}

MAIN_PID="$$"
echo "timestamp,mem_available_kib" > "${LOG_DIR}/${NAME}.resources.csv"
monitor &
MONITOR_PID="$!"
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

# Batch 20 was selected after batch 24 reached the 20 GiB guard floor during
# its first allocation. It materially improves throughput while retaining a
# substantial unified-memory margin. Do not use fractional AutoBatch on GB10: its
# probing process OOMs before it can choose a safe result.
taskset -c 0-16 nice -n 5 ionice -c 2 -n 4 \
  uv run --no-sync yolo detect train \
    model="${ROOT_DIR}/models/pretrained/yolo26l.pt" \
    data="${ROOT_DIR}/data/plushie/weekend_grouped_holdout.yaml" \
    epochs=300 patience=50 batch=20 imgsz=1280 cache=disk device=0 workers=14 \
    save=True save_period=1 project="${RUN_ROOT}" name="${NAME}" exist_ok=True \
    optimizer=MuSGD lr0=0.00038 lrf=0.882 momentum=0.948 weight_decay=0.00027 \
    warmup_epochs=1.0 box=9.83 cls=0.65 dfl=0.96 close_mosaic=10 \
    mosaic=0.992 mixup=0.427 copy_paste=0.404 scale=0.95 translate=0.275 fliplr=0.304 \
    degrees=0.0 shear=0.0 hsv_h=0.013 hsv_s=0.353 hsv_v=0.194 amp=True plots=True
