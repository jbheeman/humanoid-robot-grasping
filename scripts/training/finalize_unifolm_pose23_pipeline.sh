#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
RUNNER_PID="${1:?usage: finalize_unifolm_pose23_pipeline.sh RUNNER_PID}"
TRAIN_STATUS="${ROOT_DIR}/runs/unifolm_plush_touch/pose23-pipeline.status.json"
WAIT_STATUS="${ROOT_DIR}/runs/diagnostics/pose23_pilots/TRAINING_WAIT.json"
FINAL_STATUS="${ROOT_DIR}/runs/diagnostics/pose23_pilots/FINALIZE.json"
EVAL_LOG="${ROOT_DIR}/logs/unifolm_pose23/evaluate.log"

mkdir -p "$(dirname "${WAIT_STATUS}")" "$(dirname "${EVAL_LOG}")"
/usr/bin/python3 "${ROOT_DIR}/scripts/wait_for_completion.py" \
    --pid "${RUNNER_PID}" \
    --marker "${TRAIN_STATUS}" \
    --output "${WAIT_STATUS}"
wait_rc=$?

training_ok=0
if (( wait_rc == 0 )); then
    "${TRAIN_ENV}/bin/python" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("complete") is True and x.get("exit_code") == 0 else 1)' \
        "${TRAIN_STATUS}" && training_ok=1
fi

eval_rc=99
if (( training_ok == 1 )); then
    eval_rc=0
    "${TRAIN_ENV}/bin/python" \
        "${ROOT_DIR}/scripts/evaluate_unifolm_pose23_pilots.py" \
        --root "${ROOT_DIR}" \
        --samples-per-source 96 >"${EVAL_LOG}" 2>&1 || eval_rc=$?
fi

temporary="${FINAL_STATUS}.partial"
printf '{"schema_version":1,"complete":true,"training_ok":%s,"wait_exit_code":%s,"evaluation_exit_code":%s,"physical_robot_authorized":false}\n' \
    "${training_ok}" "${wait_rc}" "${eval_rc}" >"${temporary}"
mv "${temporary}" "${FINAL_STATUS}"

# Exit 2 means validation correctly rejected promotion.
if (( training_ok != 1 || (eval_rc != 0 && eval_rc != 2) )); then
    exit 2
fi
