#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_DIR="${ROOT_DIR}/robot"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the robot." >&2
  exit 1
fi

"${ROOT_DIR}/scripts/dev/fetch-arm-assets.sh"
PYTHONPATH="${ROOT_DIR}/src" python3 -m object_tracking.g1_asset_builder --target robot

uv venv --allow-existing --system-site-packages "${ROBOT_DIR}/.venv"
UV_PROJECT_ENVIRONMENT="${ROBOT_DIR}/.venv" uv sync --project "${ROBOT_DIR}" --locked

"${ROBOT_DIR}/.venv/bin/python" - <<'PY'
import importlib.util

required = ("fastapi", "unitree_sdk2py", "uvicorn", "zstandard")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing robot runtime dependencies: {missing}")

for forbidden in ("torch", "ultralytics"):
    if importlib.util.find_spec(forbidden) is not None:
        raise SystemExit(f"Robot environment must not contain training dependency: {forbidden}")

depth_backends = [
    name for name in ("rclpy", "pyrealsense2") if importlib.util.find_spec(name) is not None
]
if not depth_backends:
    raise SystemExit(
        "Install either the robot ROS2 RealSense stack (rclpy) or system pyrealsense2."
    )
print("Robot runtime ready; depth backends:", ", ".join(depth_backends))
PY

echo "Robot services are installed without CUDA/training dependencies."
echo "Start disarmed with: uv run g1 robot start --client-ip <GB10_IP> --token-file <0600-file>"
