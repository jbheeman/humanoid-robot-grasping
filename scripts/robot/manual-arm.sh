#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
HARDWARE_DOMAIN_ID="${HARDWARE_DOMAIN_ID:-0}"
MODE="${1:-observe}"

case "${MODE}" in
  observe) ALLOW_MOVEMENT=0 ;;
  move) ALLOW_MOVEMENT=1 ;;
  *) echo "Usage: scripts/robot/manual-arm.sh {observe|move}" >&2; exit 2 ;;
esac

source "${ROOT_DIR}/scripts/shared/run-logging.sh"
g1_begin_run_log "${ROOT_DIR}" "robot-manual-arm"
g1_log_command "$0" "$@"
trap 'status=$?; g1_log_exit "${status}"' EXIT

if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment. Run: uv run g1 setup robot" >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
g1_source_ros "${ROOT_DIR}" foxy
g1_configure_cyclonedds manual-arm-robot "${ROBOT_INTERFACE}" \
  "${CLIENT_IP},${UNITREE_CONTROL_PEER}" "${ROS_DOMAIN_ID}"

export PYTHONPATH="${HOME}/unitree_sdk2_python:${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

args=(
  --control-mode manual
  --manual-control-profile "${MANUAL_ARM_PROFILE:-sdk2}"
  --disable-depth
  --expected-motion-mode "${EXPECTED_MOTION_MODE:-ai}"
  --hardware-interface "${HARDWARE_INTERFACE}"
  --hardware-domain-id "${HARDWARE_DOMAIN_ID}"
)
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  args+=(--allow-movement)
fi

echo "G1 manual arm bridge: mode=${MODE}, profile=${MANUAL_ARM_PROFILE:-sdk2}, ROS domain=${ROS_DOMAIN_ID}, project NIC=${ROBOT_INTERFACE}."
echo "No camera, depth, calibration, or browser server is started by this command."
"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${args[@]}"
