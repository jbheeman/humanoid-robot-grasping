#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.sim-venv/bin/python"
[[ -x "$PY" ]] || { echo "Run scripts/sim/setup.sh first." >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
if [[ "${1:-}" == "sweep" ]] && command -v taskset >/dev/null; then
  exec taskset -c 0-14 nice -n 10 "$PY" -m object_tracking.g1_sim_cli "$@"
fi
exec "$PY" -m object_tracking.g1_sim_cli "$@"
