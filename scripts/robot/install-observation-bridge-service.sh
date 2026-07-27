#!/usr/bin/env bash
set -euo pipefail

# Installs a boot-persistent, read-only bridge on the G1.  It relays the
# native RGB stream and publishes observation state to the GB10, but never
# receives movement permission.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_NAME="g1-observation-bridge.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"
SERVICE_TEMPLATE="${ROOT_DIR}/scripts/robot/${UNIT_NAME}"
SERVICE_RENDERED="$(mktemp)"
trap 'rm -f "${SERVICE_RENDERED}"' EXIT

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

# The checked-in unit has the standard Unitree checkout as a readable default.
# Render the actual checkout path at install time so a lab-specific location
# does not make the service fail only after reboot.
sed "s|/home/unitree/humanoid-robot-grasping|${ROOT_DIR}|g" \
  "${SERVICE_TEMPLATE}" > "${SERVICE_RENDERED}"
"${SUDO[@]}" install -m 0644 "${SERVICE_RENDERED}" "${UNIT_PATH}"
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable --now "${UNIT_NAME}"
"${SUDO[@]}" systemctl status "${UNIT_NAME}" --no-pager

echo
echo "The G1 observation bridge will now start after every boot."
echo "It relays RGB and publishes read-only ROS state to the configured GB10."
echo "It does not permit arm movement."
