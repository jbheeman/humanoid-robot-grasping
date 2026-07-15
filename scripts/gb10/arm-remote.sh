#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GB10_PYTHON="${GB10_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
ROBOT_HOST="${ROBOT_HOST:-192.168.0.213}"
ROS_INTERFACE="${ROS_INTERFACE:-auto}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if [[ ! -x "${GB10_PYTHON}" ]]; then
  echo "Missing GB10 environment. Run: uv run g1 setup gb10" >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
# Foxy Fast DDS repeatedly crashes while parsing Jazzy/Cyclone discovery data
# on the stock G1 image. Keep this arm-only path Fast DDS on both hosts.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
g1_source_ros "${ROOT_DIR}" jazzy
if ! ros2 pkg prefix rmw_fastrtps_cpp >/dev/null 2>&1; then
  echo "Missing ros-jazzy-rmw-fastrtps-cpp on the GB10." >&2
  echo "Install it once: sudo apt-get install ros-jazzy-rmw-fastrtps-cpp" >&2
  exit 1
fi
if [[ "${ROS_INTERFACE}" == "auto" ]]; then
  ROS_INTERFACE="$(ip -4 route get "${ROBOT_HOST}" | awk 'NR==1 {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')"
fi
if [[ -z "${ROS_INTERFACE}" ]]; then
  echo "Could not determine the GB10 interface used to reach ${ROBOT_HOST}." >&2
  exit 1
fi
g1_configure_cyclonedds manual-arm-gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${GB10_PYTHON}" -m object_tracking.manual_arm_cli "$@"
