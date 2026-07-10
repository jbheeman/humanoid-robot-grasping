#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VIEWER_DIR="${ROOT_DIR}/scripts/gb10/web"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. On the GB10, run:"
  echo "  UV_PYTHON=3.12 uv sync --only-group vision --only-group train --locked"
  exit 1
fi

MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.pt}"
UDP_PORT="${UDP_PORT:-5600}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VIEWER_PORT="${VIEWER_PORT:-8080}"
VISION_WIDTH="${VISION_WIDTH:-1280}"
VISION_HEIGHT="${VISION_HEIGHT:-720}"
VISION_FPS="${VISION_FPS:-30}"
IMGSZ="${IMGSZ:-960}"
CONF="${CONF:-0.35}"
INFER_EVERY="${INFER_EVERY:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-75}"
MAX_DET="${MAX_DET:-20}"
OPENCV_THREADS="${OPENCV_THREADS:-16}"
TORCH_THREADS="${TORCH_THREADS:-16}"
STREAM_FPS="${STREAM_FPS:-${VISION_FPS}}"
ACCESS_LOG="${ACCESS_LOG:-0}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-gst-launch}"
STOP_EXISTING="${STOP_EXISTING:-1}"
ROBOT_HOST="${ROBOT_HOST:-192.168.0.213}"
DEPTH_WS="${DEPTH_WS:-ws://${ROBOT_HOST}:8767/depth/stream}"
CALIBRATION="${CALIBRATION:-}"
ARM_URL="${ARM_URL:-http://${ROBOT_HOST}:8766}"
ARM_TOKEN_FILE="${ARM_TOKEN_FILE:-}"
ARM_HOME="${ARM_HOME:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
TARGET_HZ="${TARGET_HZ:-15}"
EXECUTE="${EXECUTE:-0}"
RESEARCH_RECORD="${RESEARCH_RECORD:-1}"
RESEARCH_HZ="${RESEARCH_HZ:-5}"
RESEARCH_ROOT="${RESEARCH_ROOT:-runs/research/arm_tracking}"
RESEARCH_LABEL="${RESEARCH_LABEL:-}"
RESEARCH_NOTES="${RESEARCH_NOTES:-}"
GB10_LAN_IP="${GB10_LAN_IP:-}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Fine-tuned plushie model not found: ${MODEL}"
  echo "Set MODEL to an existing checkpoint on the GB10."
  exit 1
fi

required_plugins=(udpsrc rtph264depay h264parse avdec_h264 videoconvert videoscale fdsink)
if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
  echo "gst-inspect-1.0 is required on the GB10. Run uv run g1 setup gb10"
  exit 1
fi
for plugin in "${required_plugins[@]}"; do
  if ! gst-inspect-1.0 "${plugin}" >/dev/null 2>&1; then
    echo "Missing GStreamer plugin: ${plugin}"
    echo "Install GStreamer base/good/bad/ugly/libav packages on the GB10."
    exit 1
  fi
done

PIPELINE="${PIPELINE:-udpsrc address=0.0.0.0 port=${UDP_PORT} buffer-size=1048576 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! queue ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=${VISION_WIDTH},height=${VISION_HEIGHT},format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

tracking_args=()
if [[ -n "${CALIBRATION}" ]]; then
  tracking_args+=(--depth-ws "${DEPTH_WS}" --calibration "${CALIBRATION}" --arm-url "${ARM_URL}" --target-hz "${TARGET_HZ}")
fi

access_log_args=()
if [[ "${ACCESS_LOG}" == "1" ]]; then
  access_log_args+=(--access-log)
fi

research_args=(
  --research-hz "${RESEARCH_HZ}"
  --research-root "${RESEARCH_ROOT}"
  --research-label "${RESEARCH_LABEL}"
  --research-notes "${RESEARCH_NOTES}"
)
if [[ "${RESEARCH_RECORD}" == "1" ]]; then
  research_args+=(--research-record)
else
  research_args+=(--no-research-record)
fi

if [[ -z "${GB10_LAN_IP}" ]] && command -v hostname >/dev/null 2>&1; then
  GB10_LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -z "${GB10_LAN_IP}" ]]; then
  GB10_LAN_IP="<GB10_IP>"
fi
DISPLAY_HOST="${HOST}"
if [[ "${DISPLAY_HOST}" == "0.0.0.0" || -z "${DISPLAY_HOST}" ]]; then
  DISPLAY_HOST="${GB10_LAN_IP}"
fi
if [[ "${EXECUTE}" == "1" ]]; then
  if [[ -z "${CALIBRATION}" || -z "${ARM_TOKEN_FILE}" ]]; then
    echo "EXECUTE=1 requires CALIBRATION and ARM_TOKEN_FILE." >&2
    exit 1
  fi
  tracking_args+=(--execute --arm-token-file "${ARM_TOKEN_FILE}")
fi
if [[ -n "${ARM_HOME}" ]]; then
  if [[ -z "${G1_ROBOT_ID}" ]]; then
    echo "ARM_HOME requires G1_ROBOT_ID to prevent cross-robot profile reuse." >&2
    exit 1
  fi
  tracking_args+=(--arm-home "${ARM_HOME}" --robot-id "${G1_ROBOT_ID}")
fi

cleanup() {
  kill "${server_pid:-}" "${viewer_pid:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

stop_processes() {
  local pattern="$1"
  local attempt

  pkill -TERM -f "${pattern}" >/dev/null 2>&1 || true
  for attempt in {1..20}; do
    if ! pgrep -f "${pattern}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  pkill -KILL -f "${pattern}" >/dev/null 2>&1 || true
}

cd "${ROOT_DIR}"

if [[ "${STOP_EXISTING}" == "1" ]]; then
  stop_processes "[o]bject_tracking.yolo_stream_server"
  stop_processes "[h]ttp.server ${VIEWER_PORT}.*gb10/web"
  stop_processes "[g]st-launch-1.0 -q udpsrc address=0.0.0.0 port=${UDP_PORT}"
fi

echo
echo "============================================================"
echo " G1 RESEARCH SERVER"
echo "============================================================"
echo "Robot RGB receiver:  udp://0.0.0.0:${UDP_PORT}"
echo "YOLO model:          ${MODEL}"
echo "Tracking profile:    ${VISION_WIDTH}x${VISION_HEIGHT} at ${VISION_FPS} FPS"
echo "Inference cadence:   every ${INFER_EVERY} frame(s)"
echo "Research recording:  ${RESEARCH_RECORD} at ${RESEARCH_HZ} Hz"
echo "Research data root:  ${ROOT_DIR}/${RESEARCH_ROOT}"
if [[ -n "${RESEARCH_LABEL}" ]]; then
  echo "Experiment label:    ${RESEARCH_LABEL}"
fi
echo
echo "ON YOUR MACBOOK (same local network), open:"
echo "  http://${DISPLAY_HOST}:${VIEWER_PORT}/unitree_dual_viewer.html?single=1"
echo
echo "If direct access is blocked, run this on the MacBook:"
echo "  ssh -N -L ${VIEWER_PORT}:127.0.0.1:${VIEWER_PORT} -L ${PORT}:127.0.0.1:${PORT} ${USER:-USER}@${DISPLAY_HOST}"
echo "Then open:"
echo "  http://127.0.0.1:${VIEWER_PORT}/unitree_dual_viewer.html?single=1"
echo
echo "Research API:"
echo "  http://${DISPLAY_HOST}:${PORT}/health"
echo "  http://${DISPLAY_HOST}:${PORT}/research/summary"
echo "  http://${DISPLAY_HOST}:${PORT}/research/export.jsonl"
echo
echo "The webpage shows RGB, hardware depth, detections, tracks,"
echo "3D object/prediction/target data, timing, IK/arm state,"
echo "rejection reasons, and research-session/export statistics."
echo "============================================================"
echo
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
  --expected-fps "${VISION_FPS}" \
  --capture-backend "${CAPTURE_BACKEND}" \
  "${access_log_args[@]}" \
  "${tracking_args[@]}" \
  "${research_args[@]}" \
  "$@" &
server_pid=$!

"${VENV_DIR}/bin/python" -m http.server "${VIEWER_PORT}" --bind "${HOST}" --directory "${VIEWER_DIR}" &
viewer_pid=$!

wait -n "${server_pid}" "${viewer_pid}"
