#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-/dev/video0}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
OUT_WIDTH="${OUT_WIDTH:-640}"
OUT_HEIGHT="${OUT_HEIGHT:-360}"

PIPELINE="${PIPELINE:-v4l2src device=${DEVICE} io-mode=2 ! image/jpeg,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! jpegdec ! videoconvert ! videoscale ! video/x-raw,width=${OUT_WIDTH},height=${OUT_HEIGHT},format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"

export PIPELINE
exec "${ROOT_DIR}/scripts/run_yolo_stream.sh" "$@"
