#!/usr/bin/env bash
set -u

gpu=${1:?GPU required}
run_id=${2:?run id required}
root=/home/aarav/Documents/g1-bunny-vla-workspace
status="$root/logs/${run_id}_gpu${gpu}.status"
failed=0

if [[ "$gpu" == 0 ]]; then
  cases=(0 1 2 3 4 10)
else
  cases=(5 6 7 8 9 11)
fi

: >"$status"
for index in "${cases[@]}"; do
  variant=head
  [[ "$index" == 10 ]] && variant=angled-left
  [[ "$index" == 11 ]] && variant=angled-right
  if "$root/scripts/run_moving_block_case.sh" "$index" "$gpu" "$variant" "$run_id"; then
    printf 'case_%02d_%s=0\n' "$index" "$variant" >>"$status"
  else
    code=$?
    printf 'case_%02d_%s=%s\n' "$index" "$variant" "$code" >>"$status"
    failed=1
  fi
done
exit "$failed"

