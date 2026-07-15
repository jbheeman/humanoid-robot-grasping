#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
PROJECT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
DEPTH_ROS_DOMAIN_ID="${G1_DEPTH_ROS_DOMAIN_ID:-43}"

if ! systemctl is-active --quiet g1-highfps-camera.service; then
  echo "The native 60 FPS camera publisher is not running." >&2
  echo "Run: sudo systemctl enable --now g1-highfps-camera.service" >&2
  exit 1
fi
if pgrep -f "${ROOT_DIR}/scripts/robot/ros_node.py" >/dev/null 2>&1; then
  echo "Another project robot node is already running." >&2
  echo "Stop the existing robot/manual-arm launcher with Ctrl+C, then retry." >&2
  exit 1
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

ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
  "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
pids+=("$!")

(
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  source "${ROOT_DIR}/scripts/shared/ros-env.sh"
  g1_source_ros "${ROOT_DIR}" foxy
  g1_configure_cyclonedds depth-robot \
    "${PROJECT_INTERFACE}" "${CLIENT_IP}" "${DEPTH_ROS_DOMAIN_ID}"
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  depth_args=(
    --depth-only
    --depth-source librealsense
    --depth-width "${DEPTH_WIDTH:-848}"
    --depth-height "${DEPTH_HEIGHT:-480}"
    --depth-capture-fps "${DEPTH_CAPTURE_FPS:-60}"
    --depth-publish-fps "${DEPTH_PUBLISH_FPS:-15}"
  )
  if [[ -n "${DEPTH_SERIAL:-}" ]]; then
    depth_args+=(--depth-serial "${DEPTH_SERIAL}")
  fi
  exec "${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${depth_args[@]}"
) &
pids+=("$!")

MANUAL_ARM_PROFILE="${MANUAL_ARM_PROFILE:-xr}" \
CLIENT_IP="${CLIENT_IP}" \
ROBOT_INTERFACE="${PROJECT_INTERFACE}" \
HARDWARE_INTERFACE="${HARDWARE_INTERFACE}" \
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER}" \
G1_PROJECT_ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  "${ROOT_DIR}/scripts/robot/manual-arm.sh" move &
pids+=("$!")

echo "Vision pointing stack started initially DISARMED."
echo "  RGB: native 960x540@60 relay to ${CLIENT_IP}:5600"
echo "  Depth: isolated CycloneDDS /g1/depth on domain ${DEPTH_ROS_DOMAIN_ID}"
echo "  Arm: isolated Fast DDS XR manual bridge on domain ${ROS_DOMAIN_ID}"

wait -n "${pids[@]}"
echo "A vision-pointing process exited; stopping the stack." >&2
exit 1
