#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id required}
root=/home/aarav/Documents/g1-bunny-vla-workspace
review="$root/artifacts/vla_dataset_review/$run_id"
log="$root/logs/${run_id}_finalize.log"
exec >"$log" 2>&1

while tmux has-session -t "${run_id}-gpu0" 2>/dev/null || tmux has-session -t "${run_id}-gpu1" 2>/dev/null; do
  sleep 10
done
source /home/aarav/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env_isaac50
cd "$root"
mkdir -p "$review"
count=0
while IFS= read -r episode; do
  case_name=$(basename "$(dirname "$episode")")
  [[ "$case_name" == rejected ]] && case_name="$(basename "$(dirname "$(dirname "$episode")")")_rejected"
  name=$(basename "$episode" .hdf5)
  python scripts/export_episode_review.py "$episode" --output-prefix "$review/${case_name}_${name}"
  count=$((count + 1))
done < <(find "datasets/$run_id" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' | sort)
if [[ "$count" -ne 12 ]]; then
  echo "expected exactly 12 accepted review episodes, found $count" >&2
  exit 3
fi
printf 'complete=1\nreview_count=%s\nreview_dir=%s\n' "$count" "$review" \
  >"$root/logs/${run_id}_finalize.status"
