#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
original_args=("$@")

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. On the GB10, run: uv run g1 setup gb10" >&2
  exit 1
fi

MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.engine}"
ROBOT_HOST="${ROBOT_HOST:-}"
ROS_INTERFACE="${ROS_INTERFACE:-auto}"
# Project DDS domain shared by the GB10 tracker and robot bridge.
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
# Normal tracking uses the robot launcher's split project process, which runs
# CycloneDDS independently from native Unitree SDK2. Keep the GB10 on the same
# RMW so the large custom depth message is discoverable and deserializable.
export RMW_IMPLEMENTATION="${G1_PROJECT_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
UDP_PORT="${UDP_PORT:-5600}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Match the validated D435I RGB profile and the robot's 60 Hz RTP relay.
# Upscaling to 1280x720 breaks pixel-space tabletop calibration and adds
# avoidable decode/resize latency before inference.
VISION_WIDTH="${VISION_WIDTH:-960}"
VISION_HEIGHT="${VISION_HEIGHT:-540}"
VISION_FPS="${VISION_FPS:-60}"
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
INTERCEPT_CONFIG="${INTERCEPT_CONFIG:-}"
ALLOW_NOMINAL_SUPPORT_PLANE="${ALLOW_NOMINAL_SUPPORT_PLANE:-0}"
ARM_HOME="${ARM_HOME:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
TARGET_HZ="${TARGET_HZ:-20}"
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
server_args=()

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Usage: scripts/gb10/start.sh [server options]"
      exit 0
      ;;
    *) server_args+=("$1") ;;
  esac
  shift
done

source "${ROOT_DIR}/scripts/shared/run-logging.sh"
log_component="gb10-vision"
g1_begin_run_log "${ROOT_DIR}" "${log_component}"
g1_log_command "$0" "${original_args[@]}"
server_launched=0
finish_log() {
  local status=$?
  g1_log_exit "${status}"
  if [[ "${status}" != "0" && "${server_launched}" == "0" ]]; then
    g1_console_error "GB10 startup failed. Details: ${G1_ACTIVE_LOG_FILE}"
  fi
}
trap finish_log EXIT

if [[ -z "${ROBOT_HOST}" ]]; then
  echo "ROBOT_HOST (the robot address) is required." >&2
  exit 1
fi
if [[ "${ROS_INTERFACE}" == "auto" ]]; then
  # The GB10 has several live NICs. DDS autodetection can choose a
  # management or private-network interface, leaving the depth subscriber
  # unable to discover the G1 even though RGB UDP still works.
  ROS_INTERFACE="$(
    ip -4 route get "${ROBOT_HOST}" 2>/dev/null |
      awk 'NR==1 {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}'
  )"
fi
if [[ -z "${ROS_INTERFACE}" ]]; then
  echo "Could not determine the GB10 interface used to reach ${ROBOT_HOST}." >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Fine-tuned plushie model not found: ${MODEL}" >&2
  echo "Set MODEL to an existing checkpoint on the GB10." >&2
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
GB10_ROS_DISTRO="${GB10_ROS_DISTRO:-$([[ -r /opt/ros/jazzy/setup.bash ]] && echo jazzy || echo humble)}"
g1_source_ros "${ROOT_DIR}" "${GB10_ROS_DISTRO}"
if ! ros2 pkg prefix "${RMW_IMPLEMENTATION}" >/dev/null 2>&1; then
  echo "Missing ros-${GB10_ROS_DISTRO}-${RMW_IMPLEMENTATION#rmw_} on the GB10." >&2
  exit 1
fi
g1_configure_cyclonedds gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
# A Jazzy ros2cli daemon exposes newer graph/type-description services that
# the robot's Foxy Fast DDS reader can discover but cannot safely deserialize.
# It is unnecessary for runtime and has caused intermittent std::bad_alloc.
pkill -TERM -u "$(id -u)" -f \
  "[r]os2cli.daemon.daemonize.*--ros-domain-id ${ROS_DOMAIN_ID}" 2>/dev/null || true
export ROS2CLI_NO_DAEMON=1
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PIPELINE="${PIPELINE:-udpsrc address=0.0.0.0 port=${UDP_PORT} buffer-size=1048576 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! queue ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=${VISION_WIDTH},height=${VISION_HEIGHT},format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

tracking_args=(--target-hz "${TARGET_HZ}")
if [[ -n "${CALIBRATION}" ]]; then
  tracking_args+=(--calibration "${CALIBRATION}")
fi
if [[ -n "${INTERCEPT_CONFIG}" ]]; then
  if [[ ! -f "${INTERCEPT_CONFIG}" ]]; then
    echo "INTERCEPT_CONFIG does not exist: ${INTERCEPT_CONFIG}" >&2
    exit 1
  fi
  if [[ -z "${CALIBRATION}" ]]; then
    echo "INTERCEPT_CONFIG requires CALIBRATION." >&2
    exit 1
  fi
  tracking_args+=(--intercept-config "${INTERCEPT_CONFIG}")
fi
if [[ "${ALLOW_NOMINAL_SUPPORT_PLANE}" == "1" ]]; then
  tracking_args+=(--allow-nominal-support-plane)
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
echo "Nominal plane fallback: ${ALLOW_NOMINAL_SUPPORT_PLANE}"
echo "Research recording:  ${RESEARCH_RECORD} at ${RESEARCH_HZ} Hz"
echo "Trajectory model:    ${TRAJECTORY_MODEL:-alpha-beta fallback only}"
echo "Process log:         ${G1_ACTIVE_LOG_FILE}"
echo
echo "Open the single GB10 UI:"
echo "  http://${DISPLAY_HOST}:${PORT}/"
echo "For off-LAN access, tunnel only this UI port:"
echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${USER:-USER}@${DISPLAY_HOST}"
echo "  http://127.0.0.1:${PORT}/"
echo
echo "The robot exposes no HTTP or WebSocket control ports."
echo "============================================================"
echo
g1_console "GB10 vision dashboard starting at http://${DISPLAY_HOST}:${PORT}/"

server_launched=1
set +e
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
  "${server_args[@]}"
status=$?
set -e
if [[ "${status}" == "0" || "${status}" == "130" ]]; then
  g1_console "GB10 vision dashboard stopped."
else
  g1_console_error "GB10 vision dashboard exited with code ${status}. Details: ${G1_ACTIVE_LOG_FILE}"
fi
exit "${status}"
