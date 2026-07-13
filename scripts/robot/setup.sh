#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_DIR="${ROOT_DIR}/robot"
ROS_SETUP="/opt/ros/foxy/setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the robot." >&2
  exit 1
fi
if [[ ! -r "${ROS_SETUP}" ]]; then
  echo "ROS 2 Foxy is required on the robot: missing ${ROS_SETUP}" >&2
  echo "Install ros-foxy-ros-base, ros-foxy-rmw-cyclonedds-cpp," >&2
  echo "ros-foxy-rosidl-generator-dds-idl, and python3-colcon-common-extensions." >&2
  exit 1
fi
if [[ ! -x "${SYSTEM_PYTHON}" ]]; then
  echo "The robot must provide ${SYSTEM_PYTHON} from Ubuntu 20.04." >&2
  exit 1
fi
if [[ "$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.8" ]]; then
  echo "The Foxy robot runtime requires /usr/bin/python3 to be Python 3.8." >&2
  exit 1
fi

set +u
source "${ROS_SETUP}"
set -u
bash "${ROOT_DIR}/scripts/shared/build-ros-workspaces.sh"
set +u
source "${ROOT_DIR}/.ros/foxy/unitree/setup.bash"
source "${ROOT_DIR}/.ros/foxy/project/setup.bash"
set -u

uv venv --clear --system-site-packages --python "${SYSTEM_PYTHON}" "${ROBOT_DIR}/.venv"
UV_PROJECT_ENVIRONMENT="${ROBOT_DIR}/.venv" uv sync --project "${ROBOT_DIR}" --locked

"${ROBOT_DIR}/.venv/bin/python" - <<'PY'
import importlib.util
import sys

required = ("rclpy", "unitree_api", "unitree_hg", "g1_control_interfaces", "zstandard")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing robot runtime dependencies: {missing}")
if sys.version_info[:2] != (3, 8):
    raise SystemExit(f"Robot runtime must use Python 3.8, got {sys.version.split()[0]}")

for forbidden in ("fastapi", "torch", "ultralytics", "unitree_sdk2py"):
    if importlib.util.find_spec(forbidden) is not None:
        raise SystemExit(f"Robot environment contains removed dependency: {forbidden}")

print("Robot ROS 2 runtime ready:", sys.version.split()[0])
PY

echo "Robot runtime is installed without HTTP, CUDA, or training dependencies."
echo "Start disarmed with: uv run g1 robot start --client-ip <GB10_IP>"
