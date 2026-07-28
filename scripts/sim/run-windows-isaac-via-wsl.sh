#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 REPLAY.json OUTPUT.json [isaac-g1-replay options...]" >&2
  exit 2
fi

replay=$1
output=$2
shift 2

windows_user=${ISAAC_WINDOWS_USER:-${USER}}
project_root=${ISAAC_PROJECT_ROOT:-/mnt/c/Users/$windows_user/Documents/coding/isaacsim/humanoid-robot-grasping}
unitree_root=${ISAAC_UNITREE_ROOT:-/mnt/c/Users/$windows_user/Documents/coding/isaacsim/unitree_sim_isaaclab}
python_exe=${ISAAC_PYTHON_EXE:-/mnt/c/Users/$windows_user/Documents/isaac-validation-data/venvs/isaac601-win/Scripts/python.exe}

for required in \
  "$python_exe" \
  "$project_root/scripts/sim/isaac-g1-replay.py" \
  "$unitree_root/robots/unitree.py" \
  "$replay"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required Isaac validation path: $required" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$output")"
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export WSLENV="${WSLENV:+$WSLENV:}OMNI_KIT_ACCEPT_EULA:PRIVACY_CONSENT"

"$python_exe" \
  "$(wslpath -w "$project_root/scripts/sim/isaac-g1-replay.py")" \
  "$(wslpath -w "$replay")" \
  --output "$(wslpath -w "$output")" \
  --project-root "$(wslpath -w "$project_root")" \
  --unitree-sim-root "$(wslpath -w "$unitree_root")" \
  --device cpu \
  --kit_args=--/app/vulkan=false \
  "$@"
