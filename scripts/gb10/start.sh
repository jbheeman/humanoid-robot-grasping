#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. On the GB10, run: uv run g1 setup gb10" >&2
  exit 1
fi

MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.engine}"
#ROBOT_HOST="${ROBOT_HOST:-}"
ROBOT_HOST="192.168.0.213"
ROS_INTERFACE="${ROS_INTERFACE:-auto}"
# Manual arm control runs in domain 42. Vision-pointing overrides this process
# to depth domain 43 so Foxy CycloneDDS never discovers the Fast DDS arm graph.
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
UDP_PORT="${UDP_PORT:-5600}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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
CALIBRATION="${CALIBRATION:-}"
ARM_HOME="${ARM_HOME:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
TARGET_HZ="${TARGET_HZ:-15}"
EXECUTE="${EXECUTE:-0}"
RESEARCH_RECORD="${RESEARCH_RECORD:-1}"
RESEARCH_HZ="${RESEARCH_HZ:-5}"
RESEARCH_ROOT="${RESEARCH_ROOT:-runs/research/arm_tracking}"
RESEARCH_LABEL="${RESEARCH_LABEL:-}"
RESEARCH_NOTES="${RESEARCH_NOTES:-}"
# Compact 3D GRU forecaster trained on synthetic trajectories. Override only
# to compare a new checkpoint; the launcher should exercise this by default.
TRAJECTORY_MODEL="${TRAJECTORY_MODEL:-${ROOT_DIR}/models/plushie_detector/trajectory_gru_synth_pretrain_17h/best.pt}"
GB10_LAN_IP="${GB10_LAN_IP:-}"
ARM_COMMISSIONING=0
VISION_POINTING=0
server_args=()

while (($#)); do
  case "$1" in
    --arm-commissioning) ARM_COMMISSIONING=1 ;;
    --vision-pointing) VISION_POINTING=1 ;;
    -h|--help)
      echo "Usage: scripts/gb10/start.sh [--arm-commissioning|--vision-pointing] [server options]"
      exit 0
      ;;
    *) server_args+=("$1") ;;
  esac
  shift
done

if [[ "${VISION_POINTING}" == "1" && -z "${CALIBRATION}" ]]; then
  CALIBRATION="${ROOT_DIR}/runs/localization/g1-tabletop-calibration.json"
fi
if [[ "${VISION_POINTING}" == "1" ]]; then
  # Match the existing D435I NVENC service without an unnecessary 720p
  # upscale or a 30 Hz capture cap. YOLO FPS remains a separate metric.
  VISION_WIDTH=960
  VISION_HEIGHT=540
  VISION_FPS=60
  STREAM_FPS=60
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  ROS_DOMAIN_ID="${G1_DEPTH_ROS_DOMAIN_ID:-43}"
fi

if [[ "${ARM_COMMISSIONING}" == "1" ]]; then
  # The stock robot's Foxy participant is stable with Jazzy only when both
  # sides of the manual arm channel use Fast DDS.
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  if [[ "${EXECUTE}" == "1" ]]; then
    echo "--arm-commissioning cannot be combined with EXECUTE=1 tracking." >&2
    exit 2
  fi
fi

if [[ -z "${ROBOT_HOST}" ]]; then
  echo "ROBOT_HOST (the robot address) is required." >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Fine-tuned plushie model not found: ${MODEL}" >&2
  echo "Set MODEL to an existing checkpoint on the GB10." >&2
  exit 1
fi
if [[ "${VISION_POINTING}" == "1" && ! -f "${CALIBRATION}" ]]; then
  echo "Vision pointing calibration not found: ${CALIBRATION}" >&2
  echo "Set CALIBRATION to the validated D435I camera-to-torso artifact." >&2
  exit 1
fi

required_plugins=(udpsrc rtph264depay h264parse avdec_h264 videoconvert videoscale fdsink)
if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
  echo "gst-inspect-1.0 is required on the GB10. Run uv run g1 setup gb10" >&2
  exit 1
fi
for plugin in "${required_plugins[@]}"; do
  if ! gst-inspect-1.0 "${plugin}" >/dev/null 2>&1; then
    echo "Missing GStreamer plugin: ${plugin}" >&2
    echo "Install GStreamer base/good/bad/ugly/libav packages on the GB10." >&2
    exit 1
  fi
done

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
g1_source_ros "${ROOT_DIR}" jazzy
g1_configure_cyclonedds gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PIPELINE="${PIPELINE:-udpsrc address=0.0.0.0 port=${UDP_PORT} buffer-size=1048576 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! queue ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=${VISION_WIDTH},height=${VISION_HEIGHT},format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

tracking_args=(--target-hz "${TARGET_HZ}")
if [[ "${VISION_POINTING}" == "1" ]]; then
  tracking_args+=(--ros-depth-only)
fi
if [[ -n "${CALIBRATION}" ]]; then
  tracking_args+=(--calibration "${CALIBRATION}")
fi
if [[ "${EXECUTE}" == "1" ]]; then
  if [[ -z "${CALIBRATION}" ]]; then
    echo "EXECUTE=1 requires CALIBRATION." >&2
    exit 1
  fi
  tracking_args+=(--execute)
fi
if [[ -n "${ARM_HOME}" ]]; then
  if [[ -z "${G1_ROBOT_ID}" ]]; then
    echo "ARM_HOME requires G1_ROBOT_ID to prevent cross-robot profile reuse." >&2
    exit 1
  fi
  tracking_args+=(--arm-home "${ARM_HOME}" --robot-id "${G1_ROBOT_ID}")
fi
if [[ -n "${TRAJECTORY_MODEL}" ]]; then
  if [[ ! -f "${TRAJECTORY_MODEL}" ]]; then
    echo "TRAJECTORY_MODEL does not exist: ${TRAJECTORY_MODEL}" >&2
    exit 1
  fi
  tracking_args+=(--trajectory-model "${TRAJECTORY_MODEL}")
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
  stop_processes "[g]st-launch-1.0 -q udpsrc address=0.0.0.0 port=${UDP_PORT}"
fi

echo
echo "============================================================"
echo " G1 ROS 2 RESEARCH + UI SERVER"
echo "============================================================"
echo "Robot RGB receiver:  udp://0.0.0.0:${UDP_PORT}"
echo "ROS peer:            ${ROBOT_HOST} (domain ${ROS_DOMAIN_ID})"
echo "YOLO model:          ${MODEL}"
echo "Tracking profile:    ${VISION_WIDTH}x${VISION_HEIGHT} at ${VISION_FPS} FPS"
echo "Movement requested:  ${EXECUTE} (robot safety gates still apply)"
echo "Arm commissioning:    ${ARM_COMMISSIONING}"
echo "Vision pointing:      ${VISION_POINTING} (planner ready; startup never moves the arm)"
echo "Research recording:  ${RESEARCH_RECORD} at ${RESEARCH_HZ} Hz"
echo "Trajectory model:    ${TRAJECTORY_MODEL:-alpha-beta fallback only}"
echo
echo "Open the single GB10 UI:"
echo "  http://${DISPLAY_HOST}:${PORT}/"
echo "Commissioning:"
if [[ "${ARM_COMMISSIONING}" == "1" ]]; then
  echo "  manual ROS bridge mode; use scripts/gb10/arm-remote.sh"
else
  echo "  http://${DISPLAY_HOST}:${PORT}/commissioning/"
fi
echo
echo "For off-LAN access, tunnel only this UI port:"
echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${USER:-USER}@${DISPLAY_HOST}"
echo "  http://127.0.0.1:${PORT}/"
echo
echo "The robot exposes no HTTP or WebSocket control ports."
echo "============================================================"
echo

exec "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
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
  "${server_args[@]}"
