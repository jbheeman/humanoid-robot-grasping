#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIPELINE="${PIPELINE:-udpsrc port=5600 buffer-size=1048576 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

export PIPELINE
exec "${ROOT_DIR}/scripts/run_yolo_stream.sh" "$@"
