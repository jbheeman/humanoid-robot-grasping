#!/usr/bin/env bash
set -euo pipefail

# Run direct SDK2 plushie tracking with safe defaults.
# This intentionally bypasses ROS2 commissioning and talks straight to rt/arm_sdk.

TRACKS_URL="${TRACKS_URL:-http://192.168.0.66:8000/tracks}"
WIDTH="${TRACK_WIDTH:-960}"
HEIGHT="${TRACK_HEIGHT:-540}"
# GB10 currently produces fresh YOLO tracks at roughly 40 Hz.  Polling faster
# than that only repeats stale boxes; the native SDK command loop itself runs
# at TRACK_CONTROL_RATE (50 Hz by default).
RATE="${TRACK_RATE:-40}"
GAIN_X="${TRACK_GAIN_X:-0.03}"
GAIN_Y="${TRACK_GAIN_Y:-0.03}"
MAX_STEP="${TRACK_MAX_STEP:-0.003}"
INTERFACE="${HARDWARE_INTERFACE:-eth0}"
DOMAIN_ID="${HARDWARE_DOMAIN_ID:-0}"
CLASSES="${TRACK_CLASSES:-plush,bunny}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

export UNITREE_SDK_PYTHONPATH="${UNITREE_SDK_PYTHONPATH:-$HOME/unitree_sdk2_python}"
if [[ ! -d "$UNITREE_SDK_PYTHONPATH" ]]; then
  echo "UNITREE_SDK_PYTHONPATH not found: $UNITREE_SDK_PYTHONPATH" >&2
  echo "Set UNITREE_SDK_PYTHONPATH to the robot SDK checkout or check file layout." >&2
  exit 1
fi

export PYTHONPATH="${UNITREE_SDK_PYTHONPATH}:${PYTHONPATH:-}"
# First live-tracking defaults: no start-up motion, require a confident stable
# target, and constrain shoulder motion to a small envelope around the measured
# pose.  Operators may widen these only after a supervised smoke test.
export TRACK_STARTUP_RAISE_RAD="${TRACK_STARTUP_RAISE_RAD:-0}"
export TRACK_MIN_CONFIDENCE="${TRACK_MIN_CONFIDENCE:-0.70}"
export TRACK_LOCK_FRAMES="${TRACK_LOCK_FRAMES:-12}"
export TRACK_DEADBAND_DEG="${TRACK_DEADBAND_DEG:-5}"
export TRACK_PITCH_ENVELOPE_RAD="${TRACK_PITCH_ENVELOPE_RAD:-0.03}"
export TRACK_YAW_ENVELOPE_RAD="${TRACK_YAW_ENVELOPE_RAD:-0.03}"
export TRACK_SLEW_STEP_RAD="${TRACK_SLEW_STEP_RAD:-0.0005}"

exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/direct_plushie_track.py" \
  --tracks-url "$TRACKS_URL" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --rate "$RATE" \
  --gain-x "$GAIN_X" \
  --gain-y "$GAIN_Y" \
  --max-step-rad "$MAX_STEP" \
  --class "$CLASSES" \
  --interface "$INTERFACE" \
  --domain-id "$DOMAIN_ID"
