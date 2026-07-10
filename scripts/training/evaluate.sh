#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

MODEL="${MODEL:-models/plushie_detector/yolo11x_plushie/weights/best.pt}"
DATA="${DATA:-data/plushie/plushie.yaml}"
IMGSZ="${IMGSZ:-960}"
PROJECT="${PROJECT:-models/plushie_detector_eval}"
NAME="${NAME:-yolo11x_plushie_val}"

if [[ -x "${VENV_DIR}/bin/yolo" ]]; then
  YOLO_BIN="${VENV_DIR}/bin/yolo"
elif command -v yolo >/dev/null 2>&1; then
  YOLO_BIN="$(command -v yolo)"
else
  echo "Could not find the yolo CLI. Run uv sync --only-group train --locked first."
  exit 1
fi

exec "${YOLO_BIN}" detect val \
  model="${MODEL}" \
  data="${DATA}" \
  imgsz="${IMGSZ}" \
  project="${PROJECT}" \
  name="${NAME}" \
  exist_ok=True
