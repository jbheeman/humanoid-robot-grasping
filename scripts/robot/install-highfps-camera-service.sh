#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_NAME="g1-highfps-camera.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

"${SUDO[@]}" install -m 0644 "${ROOT_DIR}/scripts/robot/g1-highfps-camera.service" "${UNIT_PATH}"
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable --now "${UNIT_NAME}"
"${SUDO[@]}" systemctl status "${UNIT_NAME}" --no-pager

echo
echo "High-FPS camera is enabled for future boots."
echo "It stops Unitree video_hub_pc4 through mscli and publishes 960x540@60 FPS."
echo "Restore Unitree video with: sudo ${ROOT_DIR}/scripts/robot/uninstall-highfps-camera-service.sh"
