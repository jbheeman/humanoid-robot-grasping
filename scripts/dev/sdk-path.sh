#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LINK_PATH="${ROOT_DIR}/.deps/unitree_sdk2_python"

if [[ -n "${UNITREE_SDK2PY_PATH:-}" ]]; then
  CANDIDATES=("${UNITREE_SDK2PY_PATH}")
else
  CANDIDATES=(
    "${ROOT_DIR}/../unitree_sdk2_python"
    "${ROOT_DIR}/../repos/unitree_sdk2_python"
  )
fi

for candidate in "${CANDIDATES[@]}"; do
  if [[ -d "${candidate}/unitree_sdk2py" || -f "${candidate}/pyproject.toml" || -f "${candidate}/setup.py" ]]; then
    mkdir -p "$(dirname "${LINK_PATH}")"
    ln -sfn "${candidate}" "${LINK_PATH}"
    echo "Using unitree-sdk2py checkout: ${candidate}"
    echo "Linked ${LINK_PATH}"
    exit 0
  fi
done

echo "Could not find unitree_sdk2_python checkout." >&2
echo "Checked:" >&2
for candidate in "${CANDIDATES[@]}"; do
  echo "  ${candidate}" >&2
done
echo "Set UNITREE_SDK2PY_PATH=/path/to/unitree_sdk2_python if it lives somewhere else." >&2
exit 1
