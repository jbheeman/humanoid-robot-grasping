#!/usr/bin/env bash
# Start the GB10 viewer/YOLO receiver for the G1 D435I RGB-aligned stream.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ROBOT_HOST="${ROBOT_HOST:-192.168.0.213}"
export VISION_WIDTH="${VISION_WIDTH:-960}"
export VISION_HEIGHT="${VISION_HEIGHT:-540}"
export VISION_FPS="${VISION_FPS:-60}"
export STREAM_FPS="${STREAM_FPS:-60}"
export IMGSZ="${IMGSZ:-960}"

exec "${ROOT_DIR}/scripts/gb10/start.sh" "$@"
