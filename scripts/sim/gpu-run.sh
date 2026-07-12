#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.gpu-sim-venv/bin/python"
[[ -x "$PY" ]] || { echo "Run scripts/sim/gpu-setup.sh first." >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Never preallocate the GPU. The fraction is a ceiling hint; the runner also
# defaults to a conservative batch size for a 12 GB card with other workloads.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${G1_GPU_MEMORY_FRACTION:-0.50}"
export JAX_PLATFORMS=cuda
exec "$PY" -m object_tracking.g1_gpu_sim_cli "$@"
