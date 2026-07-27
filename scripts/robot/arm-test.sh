#!/usr/bin/env bash
# Small operator launcher for the repository's custom SDK2 commissioning bridge.
# It starts a mode that permits one-joint test jogs through the GB10
# commissioning page; it never sends a joint command by itself.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
G1_ROBOT_ID="${G1_ROBOT_ID:-g1-tabletop}"
MODE="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/robot/arm-test.sh preflight
  scripts/robot/arm-test.sh enable <exact-motion-mode>

preflight starts the custom SDK2 bridge read-only.
enable permits the guarded commissioning page to issue a single 0.01 rad
right-arm sign-check jog after the operator creates and enables a session.
It does not move the arm by itself.
EOF
}

case "${MODE}" in
  preflight)
    export CLIENT_IP G1_ROBOT_ID
    export ALLOW_MOVEMENT=0
    ;;
  enable)
    EXPECTED_MOTION_MODE="${2:-}"
    if [[ -z "${EXPECTED_MOTION_MODE}" ]]; then
      echo "enable requires the exact motion mode shown during preflight." >&2
      usage >&2
      exit 2
    fi
    export CLIENT_IP G1_ROBOT_ID EXPECTED_MOTION_MODE
    export ALLOW_MOVEMENT=1
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "G1 arm test bridge"
echo "  Robot label: ${G1_ROBOT_ID}"
echo "  GB10:        ${CLIENT_IP}"
echo "  Movement:    ${ALLOW_MOVEMENT}"
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  echo "  Expected mode: ${EXPECTED_MOTION_MODE}"
  echo "  Keep the robot supported, the area clear, a spotter present, and the physical e-stop in hand."
fi
echo "  Test page: http://${CLIENT_IP}:8000/commissioning/"

exec "${ROOT_DIR}/scripts/robot/commission.sh"
