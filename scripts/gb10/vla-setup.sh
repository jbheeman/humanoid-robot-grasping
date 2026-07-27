#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GB10_PYTHON="${GB10_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
VLA_ROOT="${VLA_ROOT:-${ROOT_DIR}/.deps/unifolm-vla}"
VLA_REVISION="${VLA_REVISION:-ff6c39aeb0454cfb95418c66aef40ca777f935c1}"
HF_REVISION="${HF_REVISION:-06fee5014922ba6791cfe48d1e4aefac995dd8a2}"
MODEL_ROOT="${ROOT_DIR}/models/pretrained"
VLA_MODEL="${MODEL_ROOT}/UnifoLM-VLA-Base"
VLM_MODEL="${MODEL_ROOT}/UnifoLM-VLM-Base"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This setup is intended for the ARM64 GB10, not $(uname -m)." >&2
  exit 2
fi
if [[ ! -x "${GB10_PYTHON}" ]]; then
  echo "Missing ${GB10_PYTHON}. Run scripts/gb10/setup.sh first." >&2
  exit 1
fi
if [[ ! -d "${VLA_ROOT}/.git" ]]; then
  mkdir -p "${ROOT_DIR}/.deps"
  git clone https://github.com/unitreerobotics/unifolm-vla.git "${VLA_ROOT}"
  git -C "${VLA_ROOT}" checkout --detach "${VLA_REVISION}"
fi
if [[ "$(git -C "${VLA_ROOT}" rev-parse HEAD)" != "${VLA_REVISION}" ]]; then
  echo "UnifoLM source is not at the validated revision ${VLA_REVISION}: ${VLA_ROOT}" >&2
  echo "Preserving the existing checkout; set VLA_ROOT to a clean pinned clone." >&2
  exit 1
fi

mkdir -p "${VLA_MODEL}/checkpoints" "${VLM_MODEL}"
download() {
  local url="$1"
  local output="$2"
  local expected_size="$3"
  if [[ -f "${output}" ]] && [[ "$(stat -c '%s' "${output}")" == "${expected_size}" ]]; then
    return 0
  fi
  curl -L --fail --continue-at - --output "${output}" "${url}"
  if [[ "$(stat -c '%s' "${output}")" != "${expected_size}" ]]; then
    echo "Incomplete download: ${output}" >&2
    exit 1
  fi
}

download \
  "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base/resolve/${HF_REVISION}/checkpoints/pytorch_model.pt" \
  "${VLA_MODEL}/checkpoints/pytorch_model.pt" \
  18982885927
download \
  "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base/resolve/${HF_REVISION}/config.yaml" \
  "${VLA_MODEL}/config.yaml" \
  2156
download \
  "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base/resolve/${HF_REVISION}/dataset_statistics.json" \
  "${VLA_MODEL}/dataset_statistics.json" \
  110442

if [[ ! -f "${VLM_MODEL}/model.safetensors.index.json" ]]; then
  echo "UnifoLM-VLM-Base is required. Download it with:" >&2
  echo "  ${GB10_PYTHON%/python}/hf download unitreerobotics/UnifoLM-VLM-Base --local-dir ${VLM_MODEL}" >&2
  exit 1
fi

expected="3a82a5ce5494ce85a3c5dca1f381195520f0f60824aafbbbe56ccf8d39f21f33"
actual="$(sha256sum "${VLA_MODEL}/checkpoints/pytorch_model.pt" | awk '{print $1}')"
if [[ "${actual}" != "${expected}" ]]; then
  echo "UnifoLM-VLA checkpoint checksum mismatch: ${actual}" >&2
  exit 1
fi

"${GB10_PYTHON}" -m pip install --disable-pip-version-check --no-deps -e "${VLA_ROOT}"
"${GB10_PYTHON}" -c 'import torch, transformers, tensorflow, qwen_vl_utils, omegaconf, unifolm_vla; assert torch.cuda.is_available()'

echo "UnifoLM-VLA Base is ready on the GB10."
echo "Start the robot and GB10 vision launchers, then run: scripts/gb10/vla-chat.sh"
