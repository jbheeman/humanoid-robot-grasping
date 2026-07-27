#!/usr/bin/env bash
set -euo pipefail

workspace=/home/aarav/Documents/g1-bunny-vla-workspace
run_id="${RUN_ID:-v29-67real-motion-mixed-4k}"
final_step="${MAX_TRAIN_STEPS:-4000}"
config="${workspace}/configs/vla/v29_67real_motion_history.yaml"
data="${workspace}/datasets/plush_touch_rlds_future1_v29_67real"
relative_stats="${workspace}/datasets/plush_touch_canonical_v29_67real/SHARED_TRAIN_STATS_75_REAL_RELATIVE_POSE23_FUTURE1.json"
python=/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python

cd "${workspace}"
scripts/training/run_unifolm_v29_67real.sh smoke
scripts/training/run_unifolm_v29_67real.sh full
exec "${python}" scripts/training/evaluate_unifolm_run.py \
  --root "${workspace}" \
  --run-id "${run_id}" \
  --config "${config}" \
  --data-root "${data}" \
  --statistics "${relative_stats}" \
  --expected-final-step "${final_step}" \
  --samples-per-source 96 \
  --real-weight 0.75 \
  --gpus 0 1
