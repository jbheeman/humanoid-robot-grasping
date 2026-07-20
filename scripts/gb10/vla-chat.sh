#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GB10_PYTHON="${GB10_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
VLA_ROOT="${VLA_ROOT:-${ROOT_DIR}/.deps/unifolm-vla}"
if [[ ! -x "${GB10_PYTHON}" ]]; then
  echo "Missing GB10 environment. Run scripts/gb10/setup.sh first." >&2
  exit 1
fi
if [[ ! -d "${VLA_ROOT}/src/unifolm_vla" ]]; then
  echo "Missing official UnifoLM source. Run scripts/gb10/vla-setup.sh first." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${VLA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
ROBOT_HOST="${ROBOT_HOST:-192.168.0.213}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROS_INTERFACE="${ROS_INTERFACE:-$(ip -4 route get "${ROBOT_HOST}" | awk 'NR==1 {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')}"
export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
source "${ROOT_DIR}/scripts/shared/ros-env.sh"
g1_source_ros "${ROOT_DIR}" jazzy
g1_configure_cyclonedds vla-gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
pkill -TERM -u "$(id -u)" -f \
  "[r]os2cli.daemon.daemonize.*--ros-domain-id ${ROS_DOMAIN_ID}" 2>/dev/null || true
export ROS2CLI_NO_DAEMON=1
cd "${ROOT_DIR}"
exec "${GB10_PYTHON}" -m object_tracking.unifolm_vla_cli "$@"
