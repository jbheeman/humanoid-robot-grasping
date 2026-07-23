#!/usr/bin/env bash
set -uo pipefail

# Event-driven post-training validation. The runner PID is monitored through
# pidfd by wait_for_completion.py, so this script does not poll or sleep.

ROOT_DIR="${ROOT_DIR:-/home/aarav/Documents/g1-bunny-vla-workspace}"
TRAIN_ENV="${TRAIN_ENV:-/home/aarav/miniconda3/envs/g1-unifolm-train}"
RUNNER_PID="${1:?usage: finalize_unifolm_dynamic_pilots.sh RUNNER_PID}"
TRAIN_STATUS="${ROOT_DIR}/runs/unifolm_plush_touch/dynamic-pilots.status.json"
WAIT_STATUS="${ROOT_DIR}/runs/diagnostics/dynamic_pilots/TRAINING_WAIT.json"
FINAL_STATUS="${ROOT_DIR}/runs/diagnostics/dynamic_pilots/FINALIZE.json"
EVAL_LOG="${ROOT_DIR}/logs/unifolm_dynamic_pilots/evaluate.log"

mkdir -p "$(dirname "${FINAL_STATUS}")" "$(dirname "${EVAL_LOG}")"
rm -f "${FINAL_STATUS}"

wait_rc=0
"${TRAIN_ENV}/bin/python" "${ROOT_DIR}/scripts/wait_for_completion.py" \
    --pid "${RUNNER_PID}" \
    --marker "${TRAIN_STATUS}" \
    --output "${WAIT_STATUS}" || wait_rc=$?

training_ok=0
if [[ "${wait_rc}" -eq 0 ]]; then
    "${TRAIN_ENV}/bin/python" -c \
        'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("complete") is True and all(code == 0 for code in value.get("exit_codes", {}).values()) else 1)' \
        "${TRAIN_STATUS}" && training_ok=1
fi

eval_rc=99
if [[ "${training_ok}" -eq 1 ]]; then
    eval_rc=0
    CUDA_VISIBLE_DEVICES=0 "${TRAIN_ENV}/bin/python" \
        "${ROOT_DIR}/scripts/evaluate_unifolm_dynamic_pilots.py" \
        --root "${ROOT_DIR}" \
        --gpu 0 > "${EVAL_LOG}" 2>&1 || eval_rc=$?
fi

temporary="${FINAL_STATUS}.partial"
printf '{"schema_version":1,"complete":true,"training_ok":%s,"wait_exit_code":%s,"evaluation_exit_code":%s,"physical_robot_authorized":false}\n' \
    "${training_ok}" "${wait_rc}" "${eval_rc}" > "${temporary}"
mv "${temporary}" "${FINAL_STATUS}"

# Exit 2 is an expected validation-gate rejection, not an orchestration error.
if [[ "${training_ok}" -ne 1 || ( "${eval_rc}" -ne 0 && "${eval_rc}" -ne 2 ) ]]; then
    exit 2
fi
