#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "Install GStreamer on the GB10 first:"
  echo "  sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav"
  exit 1
fi
if [[ ! -r "${ROS_SETUP}" ]]; then
  echo "ROS 2 Jazzy is required on the GB10: missing ${ROS_SETUP}"
  echo "Install ros-jazzy-ros-base, ros-jazzy-rmw-cyclonedds-cpp,"
  echo "ros-jazzy-rosidl-generator-dds-idl, and python3-colcon-common-extensions."
  exit 1
fi
if [[ "$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "The Jazzy GB10 runtime requires /usr/bin/python3 to be Python 3.12."
  exit 1
fi

set +u
source "${ROS_SETUP}"
set -u
if ! ros2 pkg prefix rmw_fastrtps_cpp >/dev/null 2>&1; then
  echo "Install the Fast DDS RMW used by cross-version manual arm control:"
  echo "  sudo apt-get install ros-jazzy-rmw-fastrtps-cpp"
  exit 1
fi
bash "${ROOT_DIR}/scripts/shared/build-ros-workspaces.sh"
set +u
source "${ROOT_DIR}/.ros/jazzy/unitree/setup.bash"
source "${ROOT_DIR}/.ros/jazzy/project/setup.bash"
set -u

echo "Creating GB10 ROS 2 Jazzy/Python 3.12 vision environment"
uv venv --clear --system-site-packages --python "${SYSTEM_PYTHON}" "${VENV_DIR}"

echo "Installing FastAPI, YOLO/CUDA, calibration, and G1 IK runtime"
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync \
  --only-group vision --only-group train --only-group arm --locked

"${ROOT_DIR}/scripts/dev/fetch-arm-assets.sh"
"${VENV_DIR}/bin/g1" assets build --target gb10

"${VENV_DIR}/bin/python" - <<'PY'
import cv2
import fastapi
import torch
import ultralytics
import casadi
import pinocchio
import rclpy
import unitree_api
import unitree_hg
import g1_control_interfaces

print("OpenCV:", cv2.__version__)
print("FastAPI:", fastapi.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)
print("CasADi:", casadi.__version__)
print("Pinocchio:", pinocchio.__version__)
print("ROS client:", rclpy.__file__)
print("Unitree/project ROS interfaces: ready")
PY

echo "Auditing G1 29-DOF URDF joint order and limits"
"${VENV_DIR}/bin/g1-tune" joint-audit

echo
echo "GB10 vision/UI and ROS 2 environment is ready."
echo "Next: uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run"
