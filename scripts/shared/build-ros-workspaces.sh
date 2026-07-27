#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-${ROOT_DIR}/.deps/unitree_ros2}"
UNITREE_ROS_REV="12c080cb91ee55854358ee9413c2e36e543c36ee"
UNITREE_ROS_URL="https://github.com/unitreerobotics/unitree_ros2.git"
ROS_DISTRO="${ROS_DISTRO:-}"

if [[ -z "${ROS_DISTRO}" ]]; then
  echo "Source the required ROS 2 distribution before building workspaces." >&2
  exit 1
fi
if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is required (install python3-colcon-common-extensions)." >&2
  exit 1
fi
if ! ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1; then
  echo "rmw_cyclonedds_cpp is required for ROS 2 ${ROS_DISTRO}." >&2
  exit 1
fi

if [[ ! -d "${UNITREE_ROS_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UNITREE_ROS_DIR}")"
  git clone --filter=blob:none "${UNITREE_ROS_URL}" "${UNITREE_ROS_DIR}"
fi
if [[ -n "$(git -C "${UNITREE_ROS_DIR}" status --short)" ]]; then
  echo "Refusing to replace modified Unitree ROS 2 checkout: ${UNITREE_ROS_DIR}" >&2
  exit 1
fi
git -C "${UNITREE_ROS_DIR}" fetch --depth 1 origin "${UNITREE_ROS_REV}"
git -C "${UNITREE_ROS_DIR}" checkout --detach "${UNITREE_ROS_REV}"

UNITREE_PREFIX="${ROOT_DIR}/.ros/${ROS_DISTRO}/unitree"
PROJECT_PREFIX="${ROOT_DIR}/.ros/${ROS_DISTRO}/project"
BUILD_ROOT="${ROOT_DIR}/.ros/${ROS_DISTRO}/build"
LOG_ROOT="${ROOT_DIR}/.ros/${ROS_DISTRO}/log"

echo "Building pinned Unitree ROS 2 message packages (${UNITREE_ROS_REV})"
colcon --log-base "${LOG_ROOT}/unitree" build \
  --base-paths "${UNITREE_ROS_DIR}/cyclonedds_ws/src" \
  --build-base "${BUILD_ROOT}/unitree" \
  --install-base "${UNITREE_PREFIX}" \
  --packages-select unitree_api unitree_hg \
  --event-handlers console_cohesion+

set +u
source "${UNITREE_PREFIX}/setup.bash"
set -u

echo "Building project ROS 2 interfaces"
colcon --log-base "${LOG_ROOT}/project" build \
  --base-paths "${ROOT_DIR}/ros_ws/src" \
  --build-base "${BUILD_ROOT}/project" \
  --install-base "${PROJECT_PREFIX}" \
  --symlink-install \
  --event-handlers console_cohesion+

echo "ROS 2 ${ROS_DISTRO} workspaces installed under ${ROOT_DIR}/.ros/${ROS_DISTRO}."
