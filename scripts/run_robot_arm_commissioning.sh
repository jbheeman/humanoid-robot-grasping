#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
ARM_HOST="${ARM_HOST:-127.0.0.1}"
ARM_PORT="${ARM_PORT:-8766}"
ARM_TOKEN_FILE="${ARM_TOKEN_FILE:-}"
ALLOW_MOVEMENT="${ALLOW_MOVEMENT:-0}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
COMMISSIONING_ROOT="${COMMISSIONING_ROOT:-${ROOT_DIR}/runs/research/arm_commissioning}"
COMMISSIONING_PROFILE="${COMMISSIONING_PROFILE:-${HOME}/.config/g1-grasping/right-arm-home.json}"
COMMISSIONING_ACK="${COMMISSIONING_ACK:-}"

if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment: ${ROBOT_PYTHON}" >&2
  echo "Run ./scripts/setup_robot_grasping_services.sh first." >&2
  exit 1
fi
if [[ -z "${ARM_TOKEN_FILE}" || ! -f "${ARM_TOKEN_FILE}" ]]; then
  echo "ARM_TOKEN_FILE must name a mode-0600 bearer-token file." >&2
  exit 1
fi
if [[ "$(stat -c '%a' "${ARM_TOKEN_FILE}")" != "600" ]]; then
  echo "ARM_TOKEN_FILE must have permissions 0600." >&2
  exit 1
fi
if [[ -z "${G1_ROBOT_ID}" ]]; then
  echo "G1_ROBOT_ID is required so saved poses remain robot-specific." >&2
  exit 1
fi
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  if [[ -z "${EXPECTED_MOTION_MODE}" ]]; then
    echo "EXPECTED_MOTION_MODE is required for movement." >&2
    exit 1
  fi
  if [[ "${COMMISSIONING_ACK}" != "I HAVE A SPOTTER AND PHYSICAL E-STOP" ]]; then
    echo "Set COMMISSIONING_ACK='I HAVE A SPOTTER AND PHYSICAL E-STOP' to authorize startup." >&2
    exit 1
  fi
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --host "${ARM_HOST}"
  --port "${ARM_PORT}"
  --control-mode commissioning
  --token-file "${ARM_TOKEN_FILE}"
  --commissioning-root "${COMMISSIONING_ROOT}"
  --commissioning-profile "${COMMISSIONING_PROFILE}"
  --robot-id "${G1_ROBOT_ID}"
)
if [[ -n "${EXPECTED_MOTION_MODE}" ]]; then
  args+=(--expected-motion-mode "${EXPECTED_MOTION_MODE}")
fi
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  args+=(--allow-movement)
fi

echo
echo "============================================================"
echo " G1 RIGHT-ARM COMMISSIONING"
echo "============================================================"
echo "Movement permitted: ${ALLOW_MOVEMENT}"
echo "Robot ID:           ${G1_ROBOT_ID}"
echo "Expected mode:       ${EXPECTED_MOTION_MODE:-unconfigured}"
echo "Research data:       ${COMMISSIONING_ROOT}"
echo "Promoted home:       ${COMMISSIONING_PROFILE}"
echo
echo "On the MacBook, create an SSH tunnel:"
echo "  ssh -N -L ${ARM_PORT}:127.0.0.1:${ARM_PORT} ${USER:-USER}@<ROBOT_IP>"
echo "Then open:"
echo "  http://127.0.0.1:${ARM_PORT}/commissioning/"
echo
echo "The webpage Stop button is not a physical e-stop."
echo "============================================================"
echo

exec "${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/unitree_arm_bridge.py" "${args[@]}"
