#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

exec uv run --no-sync yolo detect train \
  model="${ROOT_DIR}/models/pretrained/yolo11s.pt" \
  data="${ROOT_DIR}/models/plushie_detector/yolo11x_plushie_dataset.yaml" \
  epochs=160 \
  time=12 \
  patience=30 \
  batch=64 \
  imgsz=960 \
  save=True \
  save_period=5 \
  cache=ram \
  device=0 \
  workers=10 \
  project="${ROOT_DIR}/models/plushie_detector" \
  name=yolo11s_plushie_60fps_12h \
  exist_ok=True \
  pretrained=True \
  optimizer=AdamW \
  lr0=0.0008 \
  lrf=0.01 \
  weight_decay=0.0005 \
  warmup_epochs=4.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.05 \
  cos_lr=True \
  close_mosaic=20 \
  mosaic=1.0 \
  mixup=0.05 \
  copy_paste=0.10 \
  erasing=0.20 \
  degrees=4.0 \
  translate=0.10 \
  scale=0.55 \
  shear=1.5 \
  perspective=0.0003 \
  hsv_h=0.015 \
  hsv_s=0.55 \
  hsv_v=0.35 \
  fliplr=0.5 \
  amp=True \
  plots=True
