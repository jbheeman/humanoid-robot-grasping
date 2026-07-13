#!/usr/bin/env bash
set -euo pipefail

# Lab-specific robot ROS 2 launcher. Override values only if the network
# layout changes. The robot exposes no HTTP service and needs no token.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
export ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0,eth0}"
export RGB_MODE="${RGB_MODE:-30fps}"
export RGB_WIDTH="${RGB_WIDTH:-960}"
export RGB_HEIGHT="${RGB_HEIGHT:-540}"
export RGB_FPS="${RGB_FPS:-60}"
export UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

exec "${ROOT_DIR}/scripts/robot/start.sh"
