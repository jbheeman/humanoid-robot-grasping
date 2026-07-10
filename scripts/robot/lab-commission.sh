#!/usr/bin/env bash
set -euo pipefail

# One-command, robot-local commissioning launcher. `move` is the explicit
# motion-capable mode; the launch command itself is the acknowledgement.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-read-only}"

export ARM_TOKEN_FILE="${ARM_TOKEN_FILE:-${HOME}/.config/g1-arm-token}"
export G1_ROBOT_ID="${G1_ROBOT_ID:-g1-lab-01}"

case "${MODE}" in
  read-only)
    export ALLOW_MOVEMENT=0
    ;;
  move)
    export ALLOW_MOVEMENT=1
    export EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-ai}"
    ;;
  *)
    echo "Usage: $0 [read-only|move]" >&2
    exit 2
    ;;
esac

exec "${ROOT_DIR}/scripts/robot/commission.sh"
