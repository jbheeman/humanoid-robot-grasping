#!/usr/bin/env bash
set -euo pipefail

# Lab-specific robot launcher. Override any value at invocation time only if
# the network layout changes; no token value is stored here.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
export ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
export RGB_MODE="${RGB_MODE:-30fps}"
export RGB_WIDTH="${RGB_WIDTH:-960}"
export RGB_HEIGHT="${RGB_HEIGHT:-540}"
export RGB_FPS="${RGB_FPS:-60}"
export ARM_TOKEN_FILE="${ARM_TOKEN_FILE:-${HOME}/.config/g1-arm-token}"

exec "${ROOT_DIR}/scripts/robot/start.sh"
