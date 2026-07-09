#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VIEWER_DIR="${ROOT_DIR}/scripts/vision_viewer"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. On the GB10, run:"
  echo "  UV_PYTHON=3.12 uv sync --only-group vision --only-group train --locked"
  exit 1
fi

MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolov8n_plushie_mvp/weights/best.pt}"
UDP_PORT="${UDP_PORT:-5600}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VIEWER_PORT="${VIEWER_PORT:-8080}"
IMGSZ="${IMGSZ:-960}"
CONF="${CONF:-0.35}"
INFER_EVERY="${INFER_EVERY:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-60}"
MAX_DET="${MAX_DET:-20}"
OPENCV_THREADS="${OPENCV_THREADS:-16}"
TORCH_THREADS="${TORCH_THREADS:-16}"
STREAM_FPS="${STREAM_FPS:-0}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-gst-launch}"
STOP_EXISTING="${STOP_EXISTING:-1}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Fine-tuned plushie model not found: ${MODEL}"
  echo "Set MODEL to an existing checkpoint on the GB10."
  exit 1
fi

required_plugins=(udpsrc rtph264depay h264parse avdec_h264 videoconvert videoscale fdsink)
if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
  echo "gst-inspect-1.0 is required on the GB10. Run ./scripts/setup_gb10_vision_server.sh"
  exit 1
fi
for plugin in "${required_plugins[@]}"; do
  if ! gst-inspect-1.0 "${plugin}" >/dev/null 2>&1; then
    echo "Missing GStreamer plugin: ${plugin}"
    echo "Install GStreamer base/good/bad/ugly/libav packages on the GB10."
    exit 1
  fi
done

PIPELINE="${PIPELINE:-udpsrc address=0.0.0.0 port=${UDP_PORT} buffer-size=1048576 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! queue ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  kill "${server_pid:-}" "${viewer_pid:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

if [[ "${STOP_EXISTING}" == "1" ]]; then
  pkill -f "object_tracking.yolo_stream_server" >/dev/null 2>&1 || true
  pkill -f "http.server ${VIEWER_PORT}.*vision_viewer" >/dev/null 2>&1 || true
  sleep 0.5
fi

echo "GB10 Unitree receiver: udp://0.0.0.0:${UDP_PORT}"
echo "YOLO model:            ${MODEL}"
echo "Input frame rate:       source rate (unthrottled; expected 30 FPS)"
echo "Processed stream:      http://${HOST}:${PORT}/stream.mjpg"
"${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
  --camera-name main \
  --pipeline "${PIPELINE}" \
  --model "${MODEL}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --infer-every "${INFER_EVERY}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-det "${MAX_DET}" \
  --opencv-threads "${OPENCV_THREADS}" \
  --torch-threads "${TORCH_THREADS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --stream-fps "${STREAM_FPS}" \
  --capture-backend "${CAPTURE_BACKEND}" \
  "$@" &
server_pid=$!

echo "Viewer:                http://${HOST}:${VIEWER_PORT}/unitree_dual_viewer.html?single=1"
"${VENV_DIR}/bin/python" -m http.server "${VIEWER_PORT}" --bind "${HOST}" --directory "${VIEWER_DIR}" &
viewer_pid=$!

wait -n "${server_pid}" "${viewer_pid}"
