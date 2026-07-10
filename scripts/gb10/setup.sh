#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

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

echo "Creating GB10 Python 3.12 vision environment"
uv python install 3.12
uv venv --clear --python 3.12 "${VENV_DIR}"

echo "Installing FastAPI, YOLO/CUDA, calibration, and G1 IK runtime"
UV_PYTHON=3.12 uv sync --only-group vision --only-group train --only-group arm --locked

"${ROOT_DIR}/scripts/dev/fetch-arm-assets.sh"

"${VENV_DIR}/bin/python" - <<'PY'
import cv2
import fastapi
import torch
import ultralytics
import casadi
import pinocchio

print("OpenCV:", cv2.__version__)
print("FastAPI:", fastapi.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)
print("CasADi:", casadi.__version__)
print("Pinocchio:", pinocchio.__version__)
PY

echo "Auditing G1 29-DOF URDF joint order and limits"
"${VENV_DIR}/bin/g1-tune" joint-audit

echo
echo "GB10 vision environment is ready."
echo "Next: uv run g1 gb10 start"
