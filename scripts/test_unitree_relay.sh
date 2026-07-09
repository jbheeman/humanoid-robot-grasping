#!/usr/bin/env bash
set -euo pipefail

UDP_PORT="${UDP_PORT:-5600}"
TEST_SECONDS="${TEST_SECONDS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/unitree_relay_test}"

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "gst-launch-1.0 is required."
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Listening for Unitree H264 RTP on UDP port ${UDP_PORT} for ${TEST_SECONDS}s"
set +e
timeout "${TEST_SECONDS}" gst-launch-1.0 -e -q \
  udpsrc address=0.0.0.0 port="${UDP_PORT}" buffer-size=1048576 \
  ! "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000" \
  ! queue \
  ! rtph264depay \
  ! h264parse \
  ! avdec_h264 \
  ! videoconvert \
  ! jpegenc \
  ! multifilesink location="${OUTPUT_DIR}/frame%05d.jpg" max-files=5
status=$?
set -e

if [[ "${status}" != "0" && "${status}" != "124" ]]; then
  echo "GStreamer relay test failed with exit code ${status}."
  exit "${status}"
fi

shopt -s nullglob
frames=("${OUTPUT_DIR}"/frame*.jpg)
if (( ${#frames[@]} == 0 )); then
  echo "No frames received. Start the robot relay and verify CLIENT_IP and UDP port ${UDP_PORT}."
  exit 1
fi

echo "Relay works. Decoded ${#frames[@]} JPEG frame(s):"
ls -lh "${frames[@]}"
