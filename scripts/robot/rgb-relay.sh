#!/usr/bin/env bash
set -euo pipefail

CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-}}"
MULTICAST_GROUP="${MULTICAST_GROUP:-230.1.1.1}"
MULTICAST_PORT="${MULTICAST_PORT:-1720}"
CLIENT_PORT="${CLIENT_PORT:-5600}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"

if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP is required. Set it to the GB10 address, for example:"
  echo "  CLIENT_IP=192.168.0.66 $0"
  exit 1
fi

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "gst-launch-1.0 is required on the robot."
  exit 1
fi

if ! pgrep -x videohub_pc4 >/dev/null 2>&1; then
  echo "videohub_pc4 is not running. Start the Unitree camera task first."
  exit 1
fi

echo "Relaying Unitree H264 RTP without opening /dev/video4"
echo "  source: ${MULTICAST_GROUP}:${MULTICAST_PORT} on ${ROBOT_INTERFACE}"
echo "  target: ${CLIENT_IP}:${CLIENT_PORT}"

exec gst-launch-1.0 -v \
  udpsrc multicast-group="${MULTICAST_GROUP}" address=0.0.0.0 port="${MULTICAST_PORT}" \
    auto-multicast=true multicast-iface="${ROBOT_INTERFACE}" buffer-size=1048576 \
  ! "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000" \
  ! queue \
  ! udpsink host="${CLIENT_IP}" port="${CLIENT_PORT}" sync=false async=false
