#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME="g1-highfps-camera.service"
if [[ "$(id -u)" -eq 0 ]]; then SUDO=(); else SUDO=(sudo); fi

"${SUDO[@]}" systemctl disable --now "${UNIT_NAME}" || true
"${SUDO[@]}" rm -f "/etc/systemd/system/${UNIT_NAME}"
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" /unitree/sbin/mscli startservice video_hub_pc4
echo "Restored Unitree video_hub_pc4."
