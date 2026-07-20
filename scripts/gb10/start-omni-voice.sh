#!/usr/bin/env bash
set -euo pipefail

# Small, end-to-end audio model for the GB10.  This intentionally lives in a
# dedicated venv: it must not alter the YOLO or Ollama runtimes.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.omni-venv"
MODEL="${OMNI_MODEL:-Qwen/Qwen2.5-Omni-7B}"
HOST="${OMNI_HOST:-127.0.0.1}"
PORT="${OMNI_PORT:-8910}"
# GB10 shares memory among the VLA, YOLO, and this model.  The 7B omni model
# needs a sizable fixed allocation, but 30% avoids consuming the whole device.
GPU_MEMORY_UTILIZATION="${OMNI_GPU_MEMORY_UTILIZATION:-0.30}"
MAX_MODEL_LEN="${OMNI_MAX_MODEL_LEN:-8192}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  echo "Missing ${VENV_DIR}. Run the vLLM-Omni setup first." >&2
  exit 1
fi

exec "${VENV_DIR}/bin/vllm" serve "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
