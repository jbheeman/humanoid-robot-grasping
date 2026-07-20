#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_ROOT="${PROJECT_ROOT}/third_party/unifolm-vla"
PATCH_FILE="${PROJECT_ROOT}/patches/0001-Adapt-UniFoLM-VLA-training-for-GB10.patch"
UPSTREAM_URL="https://github.com/unitreerobotics/unifolm-vla.git"
UPSTREAM_REV="ff6c39aeb0454cfb95418c66aef40ca777f935c1"
LOCAL_BRANCH="gb10-local"

if [[ ! -d "${VLA_ROOT}/.git" ]]; then
  git clone "${UPSTREAM_URL}" "${VLA_ROOT}"
fi

if [[ -n "$(git -C "${VLA_ROOT}" status --porcelain)" ]]; then
  echo "Refusing to overwrite a modified UniFoLM-VLA checkout: ${VLA_ROOT}" >&2
  exit 1
fi

git -C "${VLA_ROOT}" fetch origin "${UPSTREAM_REV}"

if git -C "${VLA_ROOT}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "The GB10 UniFoLM-VLA patch is already applied."
  exit 0
fi

git -C "${VLA_ROOT}" switch -C "${LOCAL_BRANCH}" "${UPSTREAM_REV}"
git -C "${VLA_ROOT}" am "${PATCH_FILE}"

echo "UniFoLM-VLA is pinned and patched on branch ${LOCAL_BRANCH}."
