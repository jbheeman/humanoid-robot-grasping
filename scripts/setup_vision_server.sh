#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this setup script. Install uv first, then rerun:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Creating or updating uv environment at ${VENV_DIR}"
uv venv --system-site-packages --allow-existing "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Installing vision server dependencies"
uv sync --group vision --no-default-groups

echo "Removing pip OpenCV wheels so the venv uses Ubuntu system OpenCV/GStreamer"
uv pip uninstall opencv-python opencv-contrib-python opencv-python-headless || true

echo "Checking OpenCV/GStreamer"
python - <<'PY'
import sys

try:
    import cv2
except Exception as exc:
    raise SystemExit(
        "Could not import cv2. Install Ubuntu OpenCV first, for example:\n"
        "  sudo apt-get install python3-opencv gstreamer1.0-tools "
        "gstreamer1.0-plugins-good gstreamer1.0-plugins-bad "
        "gstreamer1.0-plugins-ugly gstreamer1.0-libav\n"
        f"Original error: {exc}"
    )

print("OpenCV:", cv2.__version__)
gstreamer_line = None
for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line:
        gstreamer_line = line.strip()
        print(gstreamer_line)
        break

if gstreamer_line is None or "YES" not in gstreamer_line.upper():
    raise SystemExit(
        "OpenCV imported, but GStreamer is not enabled. The vision stream server needs "
        "Ubuntu's GStreamer-enabled OpenCV, not the pip OpenCV wheel."
    )

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
except Exception as exc:
    raise SystemExit(f"Vision server dependency import failed: {exc}")

print("Vision server environment is ready.")
PY

echo
echo "Next:"
echo "  ./scripts/run_yolo_stream.sh"
