#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REV="ae6a8403e272733e9996ef59990880330496177f"
DEPS="$ROOT/.deps/unitree_mujoco"
VENV="$ROOT/.sim-venv"
UV="$ROOT/.tools/uv"

if [[ ! -d "$DEPS/.git" ]]; then
  mkdir -p "$ROOT/.deps"
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_mujoco.git "$DEPS"
fi
git -C "$DEPS" sparse-checkout init --cone
git -C "$DEPS" sparse-checkout set --skip-checks unitree_robots/g1 LICENSE
git -C "$DEPS" fetch --depth=1 origin "$REV"
git -C "$DEPS" checkout --detach "$REV"

mkdir -p "$ROOT/.tools"
if [[ ! -x "$UV" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools" sh
fi
"$UV" python install 3.12
"$UV" venv --python 3.12 --seed --clear "$VENV"
"$UV" pip install --python "$VENV/bin/python" --group "$ROOT/pyproject.toml:sim" -e "$ROOT"
"$VENV/bin/python" -m object_tracking.g1_sim_cli doctor
