#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The robot only captures and serves frames. YOLO runs on the GB10.
export MODEL=none
exec "${ROOT_DIR}/scripts/run_opencv_yolo_stream.sh" "$@"
