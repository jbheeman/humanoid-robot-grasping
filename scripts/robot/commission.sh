#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-}}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
ALLOW_MOVEMENT="${ALLOW_MOVEMENT:-0}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
CALIBRATION="${CALIBRATION:-}"
COMMISSIONING_ROOT="${COMMISSIONING_ROOT:-${ROOT_DIR}/runs/research/arm_commissioning}"
COMMISSIONING_PROFILE="${COMMISSIONING_PROFILE:-${HOME}/.config/g1-grasping/right-arm-home.json}"
DEPTH_SOURCE="${DEPTH_SOURCE:-auto}"

if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment: ${ROBOT_PYTHON}" >&2
  echo "Run uv run g1 setup robot first." >&2
  exit 1
fi
if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP (the GB10 address) is required." >&2
  exit 1
fi
if [[ -z "${G1_ROBOT_ID}" ]]; then
  echo "G1_ROBOT_ID is required so saved poses remain robot-specific." >&2
  exit 1
fi
if [[ "${ALLOW_MOVEMENT}" == "1" && -z "${EXPECTED_MOTION_MODE}" ]]; then
  echo "EXPECTED_MOTION_MODE is required for movement." >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
g1_source_ros "${ROOT_DIR}" foxy
g1_configure_cyclonedds \
  robot-commissioning "${ROBOT_INTERFACE}" \
  "${CLIENT_IP},${UNITREE_CONTROL_PEER}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --control-mode commissioning
  --robot-id "${G1_ROBOT_ID}"
  --commissioning-root "${COMMISSIONING_ROOT}"
  --commissioning-profile "${COMMISSIONING_PROFILE}"
  --depth-source "${DEPTH_SOURCE}"
  --disable-depth
)
if [[ -n "${CALIBRATION}" ]]; then
  args+=(--calibration "${CALIBRATION}")
fi
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  args+=(--allow-movement --expected-motion-mode "${EXPECTED_MOTION_MODE}")
fi

echo
echo "============================================================"
echo " G1 RIGHT-ARM COMMISSIONING (ROS 2)"
echo "============================================================"
echo "Movement permitted: ${ALLOW_MOVEMENT}"
echo "Robot ID:           ${G1_ROBOT_ID}"
echo "Expected mode:       ${EXPECTED_MOTION_MODE:-unconfigured}"
echo "ROS interface:       ${ROBOT_INTERFACE}"
echo "ROS domain:          ${ROS_DOMAIN_ID}"
echo "Research data:       ${COMMISSIONING_ROOT}"
echo "Promoted home:       ${COMMISSIONING_PROFILE}"
echo
echo "With the GB10 server running, open:"
echo "  http://${CLIENT_IP}:8000/commissioning/"
echo "For off-LAN access, tunnel only GB10 port 8000."
echo
echo "The webpage Stop button is not a physical e-stop."
echo "============================================================"
echo

exec "${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${args[@]}"
