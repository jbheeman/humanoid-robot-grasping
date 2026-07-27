#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Accuracy-first real-time candidate. Hyperparameters follow the official
# YOLO26 L recipe, while resolution and duration match this camera/dataset.
exec uv run --no-sync yolo detect train \
  model="${ROOT_DIR}/models/pretrained/yolo26l.pt" \
  data="${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_dataset.yaml" \
  epochs=100 \
  time=12 \
  patience=25 \
  batch=16 \
  imgsz=960 \
  save=True \
  save_period=3 \
  cache=ram \
  device=0 \
  workers=10 \
  project="${ROOT_DIR}/models/plushie_detector" \
  name=yolo26l_plushie_accuracy_12h \
  exist_ok=True \
  pretrained=True \
  optimizer=MuSGD \
  lr0=0.00038 \
  lrf=0.882 \
  momentum=0.948 \
  weight_decay=0.00027 \
  warmup_epochs=1.0 \
  box=9.83 \
  cls=0.65 \
  dfl=0.96 \
  close_mosaic=10 \
  mosaic=0.992 \
  mixup=0.427 \
  copy_paste=0.404 \
  scale=0.95 \
  translate=0.275 \
  fliplr=0.304 \
  degrees=0.0 \
  shear=0.0 \
  hsv_h=0.013 \
  hsv_s=0.353 \
  hsv_v=0.194 \
  amp=True \
  plots=True
