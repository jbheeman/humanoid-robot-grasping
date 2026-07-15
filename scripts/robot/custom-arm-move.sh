#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
UNITREE_SDK_PYTHONPATH="${UNITREE_SDK_PYTHONPATH:-${HOME}/unitree_sdk2_python}"
MODE="${1:-preflight}"
shift || true

if [[ "${MODE}" != "smoke" && "${MODE}" != "preflight" && "${MODE}" != "execute" ]]; then
  echo "Usage: $0 [smoke|preflight|execute] [options]" >&2
  exit 2
fi
if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot Python: ${ROBOT_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${UNITREE_SDK_PYTHONPATH}/unitree_sdk2py/__init__.py" ]]; then
  echo "Missing Unitree SDK2 Python checkout: ${UNITREE_SDK_PYTHONPATH}" >&2
  exit 1
fi

if [[ "${MODE}" != "smoke" ]]; then
  conflicts="$(pgrep -af 'ros_node.py|arm_bridge.py|commission.sh|custom-arm-move.py' | grep -v "$$" || true)"
  if [[ -n "${conflicts}" ]]; then
    echo "Refusing to start while another robot command process exists:" >&2
    echo "${conflicts}" >&2
    exit 1
  fi
fi

export PYTHONPATH="${UNITREE_SDK_PYTHONPATH}:${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/custom-arm-move.py" \
  "${MODE}" --interface eth0 --domain-id 0 --expected-motion-mode ai "$@"
