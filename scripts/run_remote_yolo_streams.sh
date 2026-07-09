#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. On the GB10, run:"
  echo "  uv sync --only-group vision --only-group train --locked"
  exit 1
fi

ROBOT_HOST="${ROBOT_HOST:-}"
if [[ -z "${ROBOT_HOST}" ]]; then
  echo "ROBOT_HOST is required, for example:"
  echo "  ROBOT_HOST=192.168.0.212 MODEL=models/plushie_detector/yolo11x_plushie/weights/best.pt $0"
  exit 1
fi

MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolo11x_plushie/weights/best.pt}"
if [[ ! -f "${MODEL}" ]]; then
  echo "YOLO model not found: ${MODEL}"
  echo "Set MODEL to a model file available on the GB10."
  exit 1
fi

ROBOT_MAIN_PORT="${ROBOT_MAIN_PORT:-8000}"
ROBOT_CHEST_PORT="${ROBOT_CHEST_PORT:-8001}"
MAIN_INPUT_URL="${MAIN_INPUT_URL:-http://${ROBOT_HOST}:${ROBOT_MAIN_PORT}/stream.mjpg}"
CHEST_INPUT_URL="${CHEST_INPUT_URL:-http://${ROBOT_HOST}:${ROBOT_CHEST_PORT}/stream.mjpg}"
HOST="${HOST:-0.0.0.0}"
MAIN_PORT="${MAIN_PORT:-8000}"
CHEST_PORT="${CHEST_PORT:-8001}"
IMGSZ="${IMGSZ:-960}"
CONF="${CONF:-0.35}"
INFER_EVERY="${INFER_EVERY:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-60}"
MAX_DET="${MAX_DET:-20}"
OPENCV_THREADS="${OPENCV_THREADS:-16}"
TORCH_THREADS="${TORCH_THREADS:-16}"
STREAM_FPS="${STREAM_FPS:-0}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-auto}"
VIEWER_PORT="${VIEWER_PORT:-8080}"
VIEWER_DIR="${ROOT_DIR}/scripts/vision_viewer"

if ! command -v gst-inspect-1.0 >/dev/null 2>&1 \
  || ! gst-inspect-1.0 souphttpsrc >/dev/null 2>&1 \
  || ! gst-inspect-1.0 multipartdemux >/dev/null 2>&1; then
  echo "The GB10 needs GStreamer's HTTP MJPEG plugins:"
  echo "  sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good"
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mjpeg_pipeline() {
  local url="$1"
  printf '%s\n' "souphttpsrc location=${url} is-live=true do-timestamp=true ! multipartdemux ! jpegdec ! videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1"
}

run_server() {
  local camera_name="$1"
  local input_url="$2"
  local port="$3"
  shift 3

  "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
    --camera-name "${camera_name}" \
    --pipeline "$(mjpeg_pipeline "${input_url}")" \
    --model "${MODEL}" \
    --imgsz "${IMGSZ}" \
    --conf "${CONF}" \
    --infer-every "${INFER_EVERY}" \
    --jpeg-quality "${JPEG_QUALITY}" \
    --max-det "${MAX_DET}" \
    --opencv-threads "${OPENCV_THREADS}" \
    --torch-threads "${TORCH_THREADS}" \
    --host "${HOST}" \
    --port "${port}" \
    --stream-fps "${STREAM_FPS}" \
    --capture-backend "${CAPTURE_BACKEND}" \
    "$@"
}

cleanup() {
  kill "${main_pid:-}" "${chest_pid:-}" "${viewer_pid:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"
echo "GB10 main inference:  ${MAIN_INPUT_URL} -> http://${HOST}:${MAIN_PORT}"
run_server main "${MAIN_INPUT_URL}" "${MAIN_PORT}" "$@" &
main_pid=$!

echo "GB10 chest inference: ${CHEST_INPUT_URL} -> http://${HOST}:${CHEST_PORT}"
run_server chest "${CHEST_INPUT_URL}" "${CHEST_PORT}" "$@" &
chest_pid=$!

echo "GB10 viewer:          http://${HOST}:${VIEWER_PORT}/unitree_dual_viewer.html"
"${VENV_DIR}/bin/python" -m http.server "${VIEWER_PORT}" --bind "${HOST}" --directory "${VIEWER_DIR}" &
viewer_pid=$!

wait -n "${main_pid}" "${chest_pid}" "${viewer_pid}"
