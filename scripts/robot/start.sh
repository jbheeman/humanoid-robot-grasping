#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-}}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-1}"
HARDWARE_DOMAIN_ID="${HARDWARE_DOMAIN_ID:-0}"
CALIBRATION="${CALIBRATION:-}"
RGB_MODE="${RGB_MODE:-unitree}"
ALLOW_MOVEMENT="${ALLOW_MOVEMENT:-0}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
DEPTH_SOURCE="${DEPTH_SOURCE:-auto}"
ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC:-/camera/camera/depth/image_rect_raw}"
ROS_CAMERA_INFO_TOPIC="${ROS_CAMERA_INFO_TOPIC:-/camera/camera/depth/camera_info}"
ROS_DEPTH_SCALE="${ROS_DEPTH_SCALE:-0.001}"
DEPTH_WIDTH="${DEPTH_WIDTH:-640}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-480}"
DEPTH_CAPTURE_FPS="${DEPTH_CAPTURE_FPS:-30}"
DEPTH_PUBLISH_FPS="${DEPTH_PUBLISH_FPS:-15}"
DEPTH_SERIAL="${DEPTH_SERIAL:-}"
# Keep rclpy separate from the native SDK2 CycloneDDS domain in this process.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP (the GB10 address) is required." >&2
  exit 1
fi
if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment. Run: uv run g1 setup robot" >&2
  exit 1
fi
if [[ "${ALLOW_MOVEMENT}" == "1" && -z "${EXPECTED_MOTION_MODE}" ]]; then
  echo "EXPECTED_MOTION_MODE is required when ALLOW_MOVEMENT=1." >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
g1_source_ros "${ROOT_DIR}" foxy
g1_configure_cyclonedds \
  robot "${ROBOT_INTERFACE}" "${CLIENT_IP},${UNITREE_CONTROL_PEER}" "${ROS_DOMAIN_ID}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
UNITREE_SDK_PYTHONPATH="${UNITREE_SDK_PYTHONPATH:-${HOME}/unitree_sdk2_python}"
if [[ ! -f "${UNITREE_SDK_PYTHONPATH}/unitree_sdk2py/__init__.py" ]]; then
  echo "Native Unitree SDK2 was not found at ${UNITREE_SDK_PYTHONPATH}." >&2
  exit 1
fi
export PYTHONPATH="${UNITREE_SDK_PYTHONPATH}:${PYTHONPATH}"
UNITREE_SDK_DDS_LIBRARY_DIR="${UNITREE_SDK_DDS_LIBRARY_DIR:-/usr/local/lib}"
if [[ ! -f "${UNITREE_SDK_DDS_LIBRARY_DIR}/libddsc.so.0" ]]; then
  echo "Native Unitree CycloneDDS library not found: ${UNITREE_SDK_DDS_LIBRARY_DIR}/libddsc.so.0" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${UNITREE_SDK_DDS_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

node_args=(
  --control-mode tracking
  --depth-source "${DEPTH_SOURCE}"
  --ros-image-topic "${ROS_IMAGE_TOPIC}"
  --ros-camera-info-topic "${ROS_CAMERA_INFO_TOPIC}"
  --ros-depth-scale "${ROS_DEPTH_SCALE}"
  --depth-width "${DEPTH_WIDTH}"
  --depth-height "${DEPTH_HEIGHT}"
  --depth-capture-fps "${DEPTH_CAPTURE_FPS}"
  --depth-publish-fps "${DEPTH_PUBLISH_FPS}"
  --hardware-interface "${HARDWARE_INTERFACE}"
  --hardware-domain-id "${HARDWARE_DOMAIN_ID}"
)
if [[ -n "${CALIBRATION}" ]]; then
  node_args+=(--calibration "${CALIBRATION}")
fi
if [[ -n "${G1_ROBOT_ID}" ]]; then
  node_args+=(--robot-id "${G1_ROBOT_ID}")
fi
if [[ -n "${DEPTH_SERIAL}" ]]; then
  node_args+=(--depth-serial "${DEPTH_SERIAL}")
fi
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  node_args+=(--allow-movement --expected-motion-mode "${EXPECTED_MOTION_MODE}")
fi

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${RGB_MODE}" in
  unitree)
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  30fps)
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" "${ROOT_DIR}/scripts/robot/rgb-30fps.sh" &
    pids+=("$!")
    ALLOW_EXTERNAL_RGB_SOURCE=1 ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
      "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  highfps-service)
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
      "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  *)
    echo "RGB_MODE must be unitree, 30fps, or highfps-service, got: ${RGB_MODE}" >&2
    exit 2
    ;;
esac

"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${node_args[@]}" &
pids+=("$!")

echo "Robot ROS 2 node started disarmed=$([[ "${ALLOW_MOVEMENT}" == "1" ]] && echo no || echo yes)."
echo "ROS domain ${ROS_DOMAIN_ID}; interface ${ROBOT_INTERFACE}; static peers ${CLIENT_IP}, ${UNITREE_CONTROL_PEER}."
echo "RGB remains RTP/UDP ${CLIENT_IP}:${CLIENT_PORT:-5600}; the robot exposes no HTTP server."
wait -n "${pids[@]}"
echo "A robot process exited; stopping the process group." >&2
exit 1
