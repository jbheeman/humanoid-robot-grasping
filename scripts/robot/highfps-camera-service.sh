#!/usr/bin/env bash
set -euo pipefail

# Root-owned systemd entrypoint.  Unitree's master service owns videohub_pc4,
# so stop it through Unitree's supported control plane before claiming the
# camera device.  This process is the sole /dev/video4 owner afterwards.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNITREE_MSCLI="${UNITREE_MSCLI:-/unitree/sbin/mscli}"
DEVICE="${RGB_DEVICE:-/dev/video4}"

"${UNITREE_MSCLI}" stopservice video_hub_pc4 || true

for _ in $(seq 1 50); do
  if ! pgrep -x videohub_pc4 >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if pgrep -x videohub_pc4 >/dev/null 2>&1; then
  echo "Unitree videohub_pc4 still owns ${DEVICE}" >&2
  exit 1
fi

exec "${ROOT_DIR}/scripts/robot/rgb-30fps.sh"
