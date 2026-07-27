#!/usr/bin/env bash
set -euo pipefail

# One-command, robot-local ROS commissioning launcher. `move` is the explicit
# motion-capable mode; the launch command itself is the acknowledgement.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-read-only}"

export CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
export ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0,eth0}"
export UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
# Project ROS must stay separate from the G1's native Unitree DDS (domain 0)
# and from lab traffic already present on the common ROS domain 1.
# Deliberately override stale shell exports from prior manual experiments.
export ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export G1_ROBOT_ID="${G1_ROBOT_ID:-g1-lab-01}"
export RGB_MODE="${RGB_MODE:-highfps-service}"

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
