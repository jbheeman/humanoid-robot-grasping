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
# Project ROS must stay separate from the G1's native Unitree DDS (domain 0).
# scripts/robot/start.sh reads this explicit project setting.
export G1_PROJECT_ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

exec "${ROOT_DIR}/scripts/robot/start.sh"
