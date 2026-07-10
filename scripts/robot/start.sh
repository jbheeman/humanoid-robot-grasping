#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
DEPTH_PYTHON="${DEPTH_PYTHON:-${ROOT_DIR}/robot/depth-venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-}}"
DEPTH_PORT="${DEPTH_PORT:-8767}"
ARM_PORT="${ARM_PORT:-8766}"
TOKEN_FILE="${ARM_TOKEN_FILE:-}"
CALIBRATION="${CALIBRATION:-}"
RGB_MODE="${RGB_MODE:-unitree}"

if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP (the GB10 address) is required." >&2
  exit 1
fi
if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment. Run: uv sync --project robot --locked" >&2
  exit 1
fi
if [[ ! -x "${DEPTH_PYTHON}" ]]; then
  DEPTH_PYTHON="${ROBOT_PYTHON}"
fi
if [[ -z "${TOKEN_FILE}" || ! -f "${TOKEN_FILE}" ]]; then
  echo "ARM_TOKEN_FILE must name a mode-0600 bearer-token file." >&2
  exit 1
fi
if [[ "$(stat -c '%a' "${TOKEN_FILE}")" != "600" ]]; then
  echo "ARM_TOKEN_FILE must have permissions 0600." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The current robot uses Python 3.8 for its existing RealSense binding and
# Python 3.12 for DDS/arm control. Prefer the dedicated depth environment.
DEPTH_PYTHONPATH="${DEPTH_PYTHONPATH:-${HOME}/.local/lib/python3.8/site-packages}"
if [[ -d "${HOME}/cyclonedds_ws/install/cyclonedds" && -z "${CYCLONEDDS_HOME:-}" ]]; then
  export CYCLONEDDS_HOME="${HOME}/cyclonedds_ws/install/cyclonedds"
fi
if [[ -n "${CYCLONEDDS_HOME:-}" ]]; then
  export CMAKE_PREFIX_PATH="${CYCLONEDDS_HOME}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="${ROOT_DIR}/robot/.venv/lib/python3.12/site-packages/unitree_sdk2py/utils/lib:${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

depth_args=(--port "${DEPTH_PORT}")
if [[ -n "${CALIBRATION}" ]]; then
  depth_args+=(--calibration "${CALIBRATION}")
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${RGB_MODE}" in
  unitree)
    CLIENT_IP="${CLIENT_IP}" "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  30fps)
    "${ROOT_DIR}/scripts/robot/rgb-30fps.sh" &
    pids+=("$!")
    ALLOW_EXTERNAL_RGB_SOURCE=1 CLIENT_IP="${CLIENT_IP}" "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  *)
    echo "RGB_MODE must be unitree or 30fps, got: ${RGB_MODE}" >&2
    exit 2
    ;;
esac
PYTHONPATH="${DEPTH_PYTHONPATH}:${PYTHONPATH}" "${DEPTH_PYTHON}" "${ROOT_DIR}/scripts/robot/depth_service.py" "${depth_args[@]}" &
pids+=("$!")
"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/arm_bridge.py" \
  --port "${ARM_PORT}" --token-file "${TOKEN_FILE}" &
pids+=("$!")

echo "Robot services started disarmed: RGB relay, depth ${DEPTH_PORT}, arm ${ARM_PORT}."
wait -n "${pids[@]}"
echo "A robot service exited; stopping the service group." >&2
exit 1
