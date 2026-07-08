#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p runs/training

MODEL="${MODEL:-yolo11x.pt}"
EPOCHS="${EPOCHS:-160}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-auto}"
CACHE="${CACHE:-auto}"
RAM_RESERVE_GB="${RAM_RESERVE_GB:-16}"
CPU_RESERVE_PERCENT="${CPU_RESERVE_PERCENT:-50}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"
PATIENCE="${PATIENCE:-40}"
NAME="${NAME:-yolo11x_plushie}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${NAME}_guarded"
LOG_PATH="runs/training/${RUN_ID}.log"
UNIT_NAME="plushie-train-${RUN_ID}"

total_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
total_gib="$((total_kib / 1024 / 1024))"
memory_max_gib="$((total_gib - RAM_RESERVE_GB))"
if (( memory_max_gib < 8 )); then
  echo "Refusing to start: MemoryMax would be ${memory_max_gib}G." >&2
  exit 1
fi

cpu_count="$(nproc)"
cpu_quota_percent="$(python3 - <<PY
cpus = int(${cpu_count})
reserve = float(${CPU_RESERVE_PERCENT})
quota = max(1.0, cpus * max(5.0, min(100.0, 100.0 - reserve)))
print(int(quota))
PY
)"

export MODEL EPOCHS IMGSZ BATCH WORKERS CACHE RAM_RESERVE_GB CPU_RESERVE_PERCENT SAVE_PERIOD PATIENCE NAME
export ALLOW_AUTOBATCH="${ALLOW_AUTOBATCH:-0}"
export FORCE_RAM_CACHE="${FORCE_RAM_CACHE:-0}"
export DRY_RUN="${DRY_RUN:-0}"

echo "Guarded plushie training launch"
echo "  log: ${LOG_PATH}"
echo "  model: ${MODEL}"
echo "  imgsz: ${IMGSZ}"
echo "  batch: ${BATCH}"
echo "  workers: ${WORKERS}"
echo "  cache: ${CACHE}"
echo "  memory cap: ${memory_max_gib}G"
echo "  swap cap: 0"
echo "  cpu quota: ${cpu_quota_percent}%"

if command -v systemd-run >/dev/null 2>&1; then
  systemd-run --user \
    --unit="${UNIT_NAME}" \
    --same-dir \
    --collect \
    -p "MemoryMax=${memory_max_gib}G" \
    -p "MemorySwapMax=0" \
    -p "CPUQuota=${cpu_quota_percent}%" \
    -p "TasksMax=infinity" \
    --setenv="MODEL=${MODEL}" \
    --setenv="EPOCHS=${EPOCHS}" \
    --setenv="IMGSZ=${IMGSZ}" \
    --setenv="BATCH=${BATCH}" \
    --setenv="WORKERS=${WORKERS}" \
    --setenv="CACHE=${CACHE}" \
    --setenv="RAM_RESERVE_GB=${RAM_RESERVE_GB}" \
    --setenv="CPU_RESERVE_PERCENT=${CPU_RESERVE_PERCENT}" \
    --setenv="SAVE_PERIOD=${SAVE_PERIOD}" \
    --setenv="PATIENCE=${PATIENCE}" \
    --setenv="NAME=${NAME}" \
    --setenv="ALLOW_AUTOBATCH=${ALLOW_AUTOBATCH}" \
    --setenv="FORCE_RAM_CACHE=${FORCE_RAM_CACHE}" \
    --setenv="DRY_RUN=${DRY_RUN}" \
    bash -lc './scripts/train_plushie_detector.sh 2>&1 | tee -a "$0"' "${LOG_PATH}"
  echo "  unit: ${UNIT_NAME}"
  echo "Follow logs:"
  echo "  tail -f ${LOG_PATH}"
  echo "Check unit:"
  echo "  systemctl --user status ${UNIT_NAME}"
else
  echo "systemd-run not available; falling back to tmux without cgroup caps." >&2
  tmux new-session -d -s "${UNIT_NAME}" "cd '${ROOT_DIR}' && ./scripts/train_plushie_detector.sh 2>&1 | tee -a '${LOG_PATH}'"
  echo "  tmux session: ${UNIT_NAME}"
fi
