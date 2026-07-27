#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
GB10_ROS_DISTRO="${GB10_ROS_DISTRO:-}"
if [[ -z "${GB10_ROS_DISTRO}" ]]; then
  if [[ -r /opt/ros/jazzy/setup.bash ]]; then
    GB10_ROS_DISTRO="jazzy"
  elif [[ -r /opt/ros/humble/setup.bash ]]; then
    GB10_ROS_DISTRO="humble"
  else
    GB10_ROS_DISTRO="jazzy"
  fi
fi
if [[ "${GB10_ROS_DISTRO}" != "jazzy" && "${GB10_ROS_DISTRO}" != "humble" ]]; then
  echo "GB10_ROS_DISTRO must be jazzy or humble." >&2
  exit 2
fi
ROS_SETUP="/opt/ros/${GB10_ROS_DISTRO}/setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"
EXPECTED_PYTHON="$([[ "${GB10_ROS_DISTRO}" == "jazzy" ]] && echo 3.12 || echo 3.10)"

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
  echo "ROS 2 ${GB10_ROS_DISTRO} is required: missing ${ROS_SETUP}"
  echo "Install ros-${GB10_ROS_DISTRO}-ros-base, ros-${GB10_ROS_DISTRO}-rmw-cyclonedds-cpp,"
  echo "ros-${GB10_ROS_DISTRO}-rosidl-generator-dds-idl, and python3-colcon-common-extensions."
  exit 1
fi
if [[ "$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "${EXPECTED_PYTHON}" ]]; then
  echo "ROS 2 ${GB10_ROS_DISTRO} requires /usr/bin/python3 ${EXPECTED_PYTHON}."
  exit 1
fi

set +u
source "${ROS_SETUP}"
set -u
if ! ros2 pkg prefix rmw_fastrtps_cpp >/dev/null 2>&1; then
  echo "Install the Fast DDS RMW used by cross-version manual arm control:"
  echo "  sudo apt-get install ros-${GB10_ROS_DISTRO}-rmw-fastrtps-cpp"
  exit 1
fi
bash "${ROOT_DIR}/scripts/shared/build-ros-workspaces.sh"
set +u
source "${ROOT_DIR}/.ros/${GB10_ROS_DISTRO}/unitree/setup.bash"
source "${ROOT_DIR}/.ros/${GB10_ROS_DISTRO}/project/setup.bash"
set -u

echo "Creating GB10 ROS 2 ${GB10_ROS_DISTRO}/Python ${EXPECTED_PYTHON} vision environment"
uv venv --clear --system-site-packages --python "${SYSTEM_PYTHON}" "${VENV_DIR}"

echo "Installing FastAPI, YOLO/CUDA, calibration, and G1 IK runtime"
if [[ "${GB10_ROS_DISTRO}" == "humble" ]]; then
  # Ubuntu 22.04 workstations commonly have CUDA 12-capable drivers. Avoid the
  # GB10's locked CUDA 13 stack and install a stable Ampere-compatible profile.
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync \
    --only-group vision --only-group arm --locked
  uv pip install --python "${VENV_DIR}/bin/python" \
    "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "${VENV_DIR}/bin/python" \
    "ultralytics>=8.2,<9" "opencv-python>=4.9" "tensorrt-cu12>=10,<11"
else
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync \
    --only-group vision --only-group train --only-group arm --locked
fi

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
