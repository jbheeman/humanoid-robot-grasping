#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-opencv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run this first:"
  echo "  ./scripts/setup_opencv_vision_server.sh"
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

if [[ "${STOP_EXISTING}" == "1" ]]; then
  pkill -f "object_tracking.yolo_stream_server" >/dev/null 2>&1 || true
  pkill -f "gst-launch-1.0 -q .*fdsink fd=1" >/dev/null 2>&1 || true
  sleep 0.5
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
  echo "Starting Unitree G1 main camera stream on http://${HOST}:${MAIN_PORT} (${MAIN_DEVICE})"
  G1_CAMERA_NAME="${MAIN_CAMERA_NAME}" G1_CAMERA_DEVICE="${MAIN_DEVICE}" \
    "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
      --camera-name "${MAIN_CAMERA_NAME}" \
      --port "${MAIN_PORT}" \
      "${common_args[@]}" \
      --capture-backend "${CAPTURE_BACKEND}" \
      "$@" &
  main_pid=$!

  echo "Starting Unitree G1 chest camera stream on http://${HOST}:${CHEST_PORT} (${CHEST_DEVICE})"
  trap 'kill "${main_pid}" >/dev/null 2>&1 || true' EXIT INT TERM
  G1_CAMERA_NAME="${CHEST_CAMERA_NAME}" G1_CAMERA_DEVICE="${CHEST_DEVICE}" \
    exec "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
      --camera-name "${CHEST_CAMERA_NAME}" \
      --port "${CHEST_PORT}" \
      "${common_args[@]}" \
      --capture-backend "${CAPTURE_BACKEND}" \
      "$@"
fi

run_stream "${MAIN_CAMERA_NAME}" "${MAIN_DEVICE}" "${PORT}" \
  "$@"
