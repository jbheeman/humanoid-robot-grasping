#!/usr/bin/env bash
set -euo pipefail

# Replacement producer for the G1 main RGB camera. It is separate from
# videohub_pc4, which captures at 15 FPS. Stop videohub_pc4 through
# `sudo /unitree/sbin/mscli stopservice video_hub_pc4` before this starts.

DEVICE="${RGB_DEVICE:-/dev/video4}"
WIDTH="${RGB_WIDTH:-1280}"
HEIGHT="${RGB_HEIGHT:-720}"
FPS="${RGB_FPS:-30}"
BITRATE="${RGB_BITRATE:-8000000}"
MULTICAST_GROUP="${MULTICAST_GROUP:-230.1.1.1}"
MULTICAST_PORT="${MULTICAST_PORT:-1720}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"

if [[ "${FPS}" != "30" && "${FPS}" != "60" ]]; then
  echo "RGB_FPS must be 30 or 60; use 30 for the 1280x720 tracking default." >&2
  exit 2
fi

if pgrep -x videohub_pc4 >/dev/null 2>&1; then
  echo "videohub_pc4 still owns ${DEVICE}. Stop only its service first:" >&2
  echo "  sudo /unitree/sbin/mscli stopservice video_hub_pc4" >&2
  exit 1
fi

echo "Publishing ${DEVICE} at ${WIDTH}x${HEIGHT} ${FPS} FPS to ${MULTICAST_GROUP}:${MULTICAST_PORT} on ${ROBOT_INTERFACE}"

exec gst-launch-1.0 -e \
  v4l2src device="${DEVICE}" ! \
  "video/x-raw,format=YUY2,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" ! \
  queue ! nvvidconv ! \
  "video/x-raw(memory:NVMM),format=NV12,width=${WIDTH},height=${HEIGHT}" ! \
  queue ! nvv4l2h264enc bitrate="${BITRATE}" iframeinterval="${FPS}" idrinterval="${FPS}" insert-sps-pps=1 ! \
  h264parse ! rtph264pay pt=96 config-interval=1 ! \
  udpsink host="${MULTICAST_GROUP}" port="${MULTICAST_PORT}" multicast-iface="${ROBOT_INTERFACE}" sync=false async=false
