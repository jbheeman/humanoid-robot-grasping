#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/.tools" "$ROOT/.deps"
if [[ ! -x "$ROOT/.tools/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools" sh
fi
"$ROOT/.tools/uv" python install 3.12
"$ROOT/.tools/uv" venv --python 3.12 --seed "$ROOT/.gpu-sim-venv"
"$ROOT/.tools/uv" pip install --python "$ROOT/.gpu-sim-venv/bin/python" \
  "jax[cuda12]==0.6.2" "mujoco==3.3.6" "mujoco-mjx==3.3.6" scipy psutil
"$ROOT/.tools/uv" pip install --python "$ROOT/.gpu-sim-venv/bin/python" -e "$ROOT"
"$ROOT/scripts/sim/gpu-run.sh" doctor
