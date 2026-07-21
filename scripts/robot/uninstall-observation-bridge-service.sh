#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME="g1-observation-bridge.service"
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

"${SUDO[@]}" systemctl disable --now "${UNIT_NAME}" || true
"${SUDO[@]}" rm -f "/etc/systemd/system/${UNIT_NAME}"
"${SUDO[@]}" systemctl daemon-reload
echo "Disabled the G1 observation bridge."
