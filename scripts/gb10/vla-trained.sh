#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${VLA_RUN_ROOT:-${ROOT_DIR}/models/unifolm_vla}"

if [[ -n "${VLA_CHECKPOINT:-}" ]]; then
  checkpoint="${VLA_CHECKPOINT}"
else
  checkpoint="$({
    find "${RUN_ROOT}" -type f -path '*/final_model/pytorch_model.pt' \
      -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -rn | sed -n '1s/^[^ ]* //p')"
fi

if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "No completed local UnifoLM checkpoint found under ${RUN_ROOT}." >&2
  echo "Set VLA_CHECKPOINT to an explicit final_model/pytorch_model.pt path." >&2
  exit 1
fi

echo "Using local trained VLA: ${checkpoint}"
exec "${ROOT_DIR}/scripts/gb10/vla-chat.sh" \
  --checkpoint "${checkpoint}" \
  --profile "${VLA_PROFILE:-g1_stack_block}" \
  "$@"
