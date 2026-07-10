#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XR_REV="7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
ROS_REV="d96d8f63ae17a7108d4f7229c00ef875ba7129c9"
DEPS_DIR="${ROOT_DIR}/.deps"
XR_DIR="${DEPS_DIR}/xr_teleoperate"
ROS_DIR="${DEPS_DIR}/unitree_ros"

fetch_checkout() {
  local url="$1"
  local revision="$2"
  local destination="$3"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  test "$(git -C "${destination}" rev-parse HEAD)" = "${revision}"
}

mkdir -p "${DEPS_DIR}"
fetch_checkout https://github.com/unitreerobotics/xr_teleoperate.git "${XR_REV}" "${XR_DIR}"
fetch_checkout https://github.com/unitreerobotics/unitree_ros.git "${ROS_REV}" "${ROS_DIR}"

test -f "${XR_DIR}/LICENSE"
test -f "${XR_DIR}/assets/g1/g1_body29_hand14.urdf"
test -f "${XR_DIR}/teleop/robot_control/robot_arm_ik.py"

echo "Pinned Unitree arm assets ready:"
echo "  xr_teleoperate ${XR_REV}"
echo "  unitree_ros    ${ROS_REV}"
