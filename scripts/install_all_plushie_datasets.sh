#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/runs/dataset_install"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/install_all_plushie_datasets.log}"
LOCK_FILE="${LOCK_FILE:-${LOG_DIR}/install_all_plushie_datasets.lock}"
WORKERS="${WORKERS:-16}"
NEGATIVE_PER_POSITIVE="${NEGATIVE_PER_POSITIVE:-1.0}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run this first:"
  echo "  uv sync --group train --no-sources"
  exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another plushie dataset install is already running."
  echo "Lock: ${LOCK_FILE}"
  pgrep -af 'object_tracking.install_plushie_datasets|install_all_plushie_datasets' || true
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "== Plushie dataset install started: $(date -Is) =="
  echo "root=${ROOT_DIR}"
  echo "workers=${WORKERS}"
  echo "negative_per_positive=${NEGATIVE_PER_POSITIVE}"
  echo

  echo "== Installing COCO 2017 teddy-bear subset + hard negatives =="
  echo "== Installing Open Images Teddy bear subset + hard negatives =="
  INSTALL_ARGS=(
    --workers "${WORKERS}"
    --negative-per-positive "${NEGATIVE_PER_POSITIVE}"
  )
  if [[ -n "${ROBOFLOW_API_KEY:-}" ]]; then
    INSTALL_ARGS+=(--roboflow-api-key "${ROBOFLOW_API_KEY}")
  fi
  "${VENV_DIR}/bin/python" -m object_tracking.install_plushie_datasets \
    "${INSTALL_ARGS[@]}"

  echo
  echo "== Roboflow public datasets note =="
  if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
    echo "Skipped: ROBOFLOW_API_KEY is not set."
    echo "Known candidates:"
    echo "  stuffed-animals-d4fei: https://universe.roboflow.com/babson-college-tvbix/stuffed-animals-d4fei"
    echo "  teddy-bear search:     https://universe.roboflow.com/search?q=class%3A%22teddy+bear%22"
    echo "Export them as YOLOv8 and merge into data/plushie, or set up an API import once keys/licenses are available."
  else
    echo "ROBOFLOW_API_KEY is set; Roboflow candidates were attempted by the Python installer."
  fi

  echo
  echo "== Final local dataset counts =="
  printf "train images: "; find data/plushie/images/train -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l
  printf "val images:   "; find data/plushie/images/val -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l
  printf "train labels: "; find data/plushie/labels/train -type f -name '*.txt' | wc -l
  printf "val labels:   "; find data/plushie/labels/val -type f -name '*.txt' | wc -l
  du -sh data/plushie data/sources 2>/dev/null || true

  echo "== Plushie dataset install finished: $(date -Is) =="
} 2>&1 | tee -a "${LOG_FILE}"
