#!/usr/bin/env bash
set -euo pipefail

# Lab-specific GB10 inference launcher. Keep access logs off: the viewer polls
# status endpoints frequently and those successful requests are not actionable.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ROBOT_HOST="${ROBOT_HOST:-192.168.123.164}"
export ROS_INTERFACE="${ROS_INTERFACE:-auto}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export MODEL="${MODEL:-${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.pt}"
export VISION_WIDTH="${VISION_WIDTH:-960}"
export VISION_HEIGHT="${VISION_HEIGHT:-540}"
export VISION_FPS="${VISION_FPS:-60}"
export STREAM_FPS="${STREAM_FPS:-60}"
export JPEG_QUALITY="${JPEG_QUALITY:-60}"
export ACCESS_LOG="${ACCESS_LOG:-0}"

export EXECUTE=0
exec "${ROOT_DIR}/scripts/gb10/start.sh"
