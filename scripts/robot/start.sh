#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-}}"
DEPTH_PORT="${DEPTH_PORT:-8767}"
ARM_PORT="${ARM_PORT:-8766}"
TOKEN_FILE="${ARM_TOKEN_FILE:-}"
CALIBRATION="${CALIBRATION:-}"

if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP (the GB10 address) is required." >&2
  exit 1
fi
if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment. Run: uv sync --project robot --locked" >&2
  exit 1
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

CLIENT_IP="${CLIENT_IP}" "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
pids+=("$!")
"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/depth_service.py" "${depth_args[@]}" &
pids+=("$!")
"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/arm_bridge.py" \
  --port "${ARM_PORT}" --token-file "${TOKEN_FILE}" &
pids+=("$!")

echo "Robot services started disarmed: RGB relay, depth ${DEPTH_PORT}, arm ${ARM_PORT}."
wait -n "${pids[@]}"
echo "A robot service exited; stopping the service group." >&2
exit 1
