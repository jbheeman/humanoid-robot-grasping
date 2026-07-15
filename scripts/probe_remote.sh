#!/usr/bin/env bash
set -euo pipefail

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "home=$HOME"
echo "python=$(command -v python || true)"
echo "conda=$(command -v conda || true)"
find "$HOME" -maxdepth 4 -type f -name 'isaac-sim.sh' -print 2>/dev/null || true
find "$HOME" -maxdepth 4 -type d \( -iname 'isaacsim*' -o -iname 'isaaclab' \) -print 2>/dev/null || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
df -h "$HOME"

