#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-opencv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run this first:"
  echo "  uv run g1 setup opencv"
  exit 1
fi

cd "${ROOT_DIR}"

MODEL="${MODEL:-none}"
PIPELINE="${PIPELINE:-}"
DUAL_STREAMS="${DUAL_STREAMS:-1}"
MAIN_CAMERA_NAME="${MAIN_CAMERA_NAME:-main}"
CHEST_CAMERA_NAME="${CHEST_CAMERA_NAME:-chest}"
MAIN_DEVICE="${MAIN_DEVICE:-videohub_pc4}"
CHEST_DEVICE="${CHEST_DEVICE:-videohub_pc4_ch}"
MAIN_STREAM_DEVICE="${MAIN_DEVICE}"
CHEST_STREAM_DEVICE="${CHEST_DEVICE}"
IMGSZ="${IMGSZ:-320}"
CONF="${CONF:-0.35}"
INFER_EVERY="${INFER_EVERY:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-60}"
MAX_DET="${MAX_DET:-20}"
OPENCV_THREADS="${OPENCV_THREADS:-16}"
TORCH_THREADS="${TORCH_THREADS:-16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAIN_PORT="${MAIN_PORT:-${PORT}}"
CHEST_PORT="${CHEST_PORT:-8001}"
STREAM_FPS="${STREAM_FPS:-0}"
STOP_EXISTING="${STOP_EXISTING:-1}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-opencv}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

resolve_camera_device() {
  local label="$1"
  local requested="$2"
  local fallback_index="$3"
  local pid
  local fd
  local target
  local configured_target
  local numeric_suffix
  local candidate

  if [[ -e "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  if [[ "${requested}" == /dev/* ]]; then
    if [[ -e "${requested}" ]]; then
      printf '%s\n' "${requested}"
      return 0
    fi
  elif [[ "${requested}" == video* ]]; then
    if [[ -e "/dev/${requested}" ]]; then
      printf '/dev/%s\n' "${requested}"
      return 0
    fi
  fi

  # Unitree's videohub names are process names, not device nodes. Prefer the
  # device actually opened by that process over deriving a video index.
  if command -v pgrep >/dev/null 2>&1; then
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      for fd in "/proc/${pid}/fd/"*; do
        [[ -e "${fd}" ]] || continue
        target="$(readlink -f "${fd}" 2>/dev/null || true)"
        if [[ "${target}" =~ ^/dev/video[0-9]+$ ]]; then
          printf '%s\n' "${target}"
          return 0
        fi
      done
      while read -r configured_target; do
        [[ -n "${configured_target}" ]] || continue
        if [[ -e "${configured_target}" ]]; then
          printf '%s\n' "${configured_target}"
          return 0
        fi
        echo "${requested} is configured for ${configured_target}, but that device does not exist." >&2
      done < <(tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null | grep -E '^/dev/video[0-9]+$' || true)
    done < <(pgrep -x "${requested}" 2>/dev/null || true)
  fi

  # Common Unitree naming pattern:
  #   videohub_pc4 -> /dev/video4 (main)
  # The chest process does not have a guaranteed numeric mapping.
  if [[ "${requested}" =~ ([0-9]+) ]]; then
    numeric_suffix="${BASH_REMATCH[1]}"
    if [[ "${requested}" != *"_ch"* ]]; then
      candidate="/dev/video${numeric_suffix}"
      if [[ -e "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  fi

  echo "Could not resolve ${label} camera device: ${requested}" >&2

  if [[ "${requested}" == *"_ch"* ]]; then
    echo "Inspect the process binding, then set CHEST_DEVICE explicitly:" >&2
    echo "  uv run g1 inspect cameras ${requested}" >&2
    return 1
  fi

  mapfile -t camera_devices < <(ls -1 /dev/video* 2>/dev/null | sort -V)
  if (( ${#camera_devices[@]} == 0 )); then
    echo "No /dev/video* devices found. Falling back to requested string: ${requested}" >&2
    echo "${requested}"
    return 0
  fi

  if (( fallback_index < ${#camera_devices[@]} )); then
    printf '%s\n' "${camera_devices[fallback_index]}"
    return 0
  fi

  printf '%s\n' "${camera_devices[0]}"
}

if [[ "${STOP_EXISTING}" == "1" ]]; then
  pkill -f "object_tracking.yolo_stream_server" >/dev/null 2>&1 || true
  pkill -f "gst-launch-1.0 -q .*fdsink fd=1" >/dev/null 2>&1 || true
  sleep 0.5
fi

if [[ -z "${PIPELINE}" ]]; then
  MAIN_STREAM_DEVICE="$(resolve_camera_device "main" "${MAIN_DEVICE}" 0)"
fi

if [[ "${DUAL_STREAMS}" == "1" && -z "${PIPELINE}" ]]; then
  CHEST_STREAM_DEVICE="$(resolve_camera_device "chest" "${CHEST_DEVICE}" 1)"
fi

args=()
if [[ -n "${PIPELINE}" ]]; then
  args+=(--pipeline "${PIPELINE}")
fi

common_args=(
  --model "${MODEL}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --infer-every "${INFER_EVERY}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-det "${MAX_DET}" \
  --opencv-threads "${OPENCV_THREADS}" \
  --torch-threads "${TORCH_THREADS}" \
  --host "${HOST}" \
  --stream-fps "${STREAM_FPS}" \
  --capture-backend "${CAPTURE_BACKEND}" \
)

run_stream() {
  local camera_name="$1"
  local device="$2"
  local port="$3"
  shift 3

  G1_CAMERA_NAME="${camera_name}" G1_CAMERA_DEVICE="${device}" \
    exec "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
      --camera-name "${camera_name}" \
      --port "${port}" \
      "${args[@]}" \
      "${common_args[@]}" \
      "$@"
}

if [[ "${DUAL_STREAMS}" == "1" && -z "${PIPELINE}" ]]; then
  echo "Starting Unitree G1 main camera stream on http://${HOST}:${MAIN_PORT} (${MAIN_STREAM_DEVICE})"
  G1_CAMERA_NAME="${MAIN_CAMERA_NAME}" G1_CAMERA_DEVICE="${MAIN_STREAM_DEVICE}" \
    "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
      --camera-name "${MAIN_CAMERA_NAME}" \
      --port "${MAIN_PORT}" \
      "${common_args[@]}" \
      --capture-backend "${CAPTURE_BACKEND}" \
      "$@" &
  main_pid=$!

  echo "Starting Unitree G1 chest camera stream on http://${HOST}:${CHEST_PORT} (${CHEST_STREAM_DEVICE})"
  trap 'kill "${main_pid}" >/dev/null 2>&1 || true' EXIT INT TERM
  G1_CAMERA_NAME="${CHEST_CAMERA_NAME}" G1_CAMERA_DEVICE="${CHEST_STREAM_DEVICE}" \
    exec "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
      --camera-name "${CHEST_CAMERA_NAME}" \
      --port "${CHEST_PORT}" \
      "${common_args[@]}" \
      --capture-backend "${CAPTURE_BACKEND}" \
      "$@"
fi

run_stream "${MAIN_CAMERA_NAME}" "${MAIN_STREAM_DEVICE}" "${PORT}" \
  "$@"
