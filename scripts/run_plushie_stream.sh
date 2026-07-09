#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run this first:"
  echo "  uv sync --group vision --group train --no-sources"
  exit 1
fi

cd "${ROOT_DIR}"

TRAINED_MODEL="${ROOT_DIR}/models/plushie_detector/yolo11x_plushie/weights/best.pt"
if [[ -z "${MODEL:-}" ]]; then
  if [[ -f "${TRAINED_MODEL}" ]]; then
    MODEL="${TRAINED_MODEL}"
  else
    MODEL="${ROOT_DIR}/models/pretrained/yolo11x.pt"
    if [[ ! -f "${MODEL}" ]]; then
      echo "Missing trained model and pretrained fallback:"
      echo "  ${TRAINED_MODEL}"
      echo "  ${MODEL}"
      echo "Run ./scripts/train_plushie_detector.sh once or set MODEL=/path/under/models."
      exit 1
    fi
  fi
fi

PIPELINE="${PIPELINE:-udpsrc port=5600 ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! rtpjitterbuffer latency=20 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=BGR ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! appsink sync=false drop=true max-buffers=1}"
IMGSZ="${IMGSZ:-960}"
CONF="${CONF:-0.35}"
INFER_EVERY="${INFER_EVERY:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-60}"
MAX_DET="${MAX_DET:-20}"
OPENCV_THREADS="${OPENCV_THREADS:-16}"
TORCH_THREADS="${TORCH_THREADS:-16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
STREAM_FPS="${STREAM_FPS:-0}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VENV_DIR}/bin/python" -m object_tracking.yolo_stream_server \
  --pipeline "${PIPELINE}" \
  --model "${MODEL}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --infer-every "${INFER_EVERY}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-det "${MAX_DET}" \
  --opencv-threads "${OPENCV_THREADS}" \
  --torch-threads "${TORCH_THREADS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --stream-fps "${STREAM_FPS}" \
  "$@"
