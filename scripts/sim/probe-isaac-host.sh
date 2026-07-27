#!/usr/bin/env bash
set -euo pipefail

printf 'host=%s\n' "$(hostname)"
printf 'kernel=%s\n' "$(uname -sr)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'home_disk='
df -h "$PWD" | awk 'NR==2 {print $4 " available"}'
printf 'memory='
free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}'
command -v git
if git lfs version >/dev/null 2>&1; then
  printf 'git_lfs=ready\n'
else
  printf 'git_lfs=missing\n'
fi
command -v docker || true
command -v conda || true

if command -v docker >/dev/null; then
  docker info --format 'docker_runtime={{json .Runtimes}}' 2>/dev/null || true
fi

for candidate in \
  "${ISAACLAB_ROOT:-}" \
  "$PWD/IsaacLab" \
  "/opt/IsaacLab" \
  "/opt/isaaclab"
do
  if [[ -n "$candidate" && -x "$candidate/isaaclab.sh" ]]; then
    printf 'isaaclab_root=%s\n' "$candidate"
    exit 0
  fi
done
printf 'isaaclab_root=missing\n'
exit 2
