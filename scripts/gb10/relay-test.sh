#!/usr/bin/env bash
set -euo pipefail

UDP_PORT="${UDP_PORT:-5600}"
TEST_SECONDS="${TEST_SECONDS:-5}"

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "gst-launch-1.0 is required."
  exit 1
fi

echo "Listening for the live Unitree H264 RTP stream on UDP port ${UDP_PORT} for ${TEST_SECONDS}s"
set +e
output="$(timeout "${TEST_SECONDS}" gst-launch-1.0 -v \
  udpsrc address=0.0.0.0 port="${UDP_PORT}" buffer-size=1048576 \
  ! "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000" \
  ! queue \
  ! rtph264depay \
  ! h264parse \
  ! avdec_h264 \
  ! videoconvert \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false fps-update-interval=1000 2>&1)"
status=$?
set -e
printf '%s\n' "${output}"

if [[ "${status}" != "0" && "${status}" != "124" ]]; then
  echo "GStreamer relay test failed with exit code ${status}."
  exit "${status}"
fi

if ! grep -qE 'video/x-raw|current:' <<< "${output}"; then
  echo "No decoded frames received. Start the robot relay and verify CLIENT_IP and UDP port ${UDP_PORT}."
  exit 1
fi

echo "Relay works. Frames were decoded in memory and discarded by fakesink; no image files were written."
