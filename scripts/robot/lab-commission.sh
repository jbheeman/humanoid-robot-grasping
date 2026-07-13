#!/usr/bin/env bash
set -euo pipefail

# One-command, robot-local ROS commissioning launcher. `move` is the explicit
# motion-capable mode; the launch command itself is the acknowledgement.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-read-only}"

export CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
export ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0,eth0}"
export UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
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
