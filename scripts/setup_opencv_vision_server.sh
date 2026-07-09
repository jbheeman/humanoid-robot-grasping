#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-opencv"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"

if [[ ! -x "${SYSTEM_PYTHON}" ]]; then
  echo "System python not found at ${SYSTEM_PYTHON}"
  echo "Set SYSTEM_PYTHON to a valid interpreter and rerun."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this setup script. Install uv first, then rerun:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Validating system OpenCV + GStreamer"
"${SYSTEM_PYTHON}" - <<'PY'
import cv2
print("OpenCV:", cv2.__version__)
print("Loaded from:", cv2.__file__)

gstreamer_line = None
for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line:
        gstreamer_line = line.strip()
        print(gstreamer_line)
        break

if gstreamer_line is None or "YES" not in gstreamer_line.upper():
    raise SystemExit(
        "OpenCV is installed but GStreamer is not enabled.\n"
        "Install / rebuild OpenCV with GStreamer support for this path."
    )
PY

cd "${ROOT_DIR}"

echo "Creating or updating uv environment at ${VENV_DIR}"
uv venv --system-site-packages --clear --python "${SYSTEM_PYTHON}" "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Installing minimal vision runtime (no train stack)"
uv pip install \
  "fastapi>=0.111,<0.112" \
  "uvicorn[standard]>=0.30,<0.31" \
  "numpy<2"

echo "Validating runtime imports"
"${VENV_DIR}/bin/python" - <<'PY'
import cv2
import fastapi
import uvicorn
print("Runtime:")
print("  OpenCV:", cv2.__version__)
print("  fastapi:", fastapi.__version__)
print("  uvicorn:", uvicorn.__version__)
PY

echo
echo "Next:"
echo "  ./scripts/run_opencv_yolo_stream.sh"
