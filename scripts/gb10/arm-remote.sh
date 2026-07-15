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
g1_source_ros "${ROOT_DIR}" jazzy
g1_configure_cyclonedds manual-arm-gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${GB10_PYTHON}" -m object_tracking.manual_arm_cli "$@"
