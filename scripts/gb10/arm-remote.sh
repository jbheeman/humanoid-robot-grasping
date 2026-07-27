#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GB10_PYTHON="${GB10_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
ROBOT_HOST="${ROBOT_HOST:-192.168.0.213}"
ROS_INTERFACE="${ROS_INTERFACE:-auto}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if [[ ! -x "${GB10_PYTHON}" ]]; then
  echo "Missing GB10 environment. Run: uv run g1 setup gb10" >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/run-logging.sh"
command_name="${1:-unknown}"
g1_begin_run_log "${ROOT_DIR}" "arm/${command_name}"
summary_printed=0
finish_arm_command() {
  local status=$?
  g1_log_exit "${status}"
  if [[ "${status}" != "0" && "${summary_printed}" == "0" ]]; then
    g1_console_error "Arm ${command_name} stopped before producing a result. Details: ${G1_ACTIVE_LOG_FILE}"
  fi
}
trap finish_arm_command EXIT
result_file="${G1_ACTIVE_LOG_FILE%.log}.result.log"
export G1_RESULT_FILE="${result_file}"
ln -sfn "$(basename "${result_file}")" \
  "$(dirname "${result_file}")/latest-result.log"

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
# Use Cyclone for the small bounded JSON arm graph.  Foxy Fast DDS has
# repeatedly aborted robot-side with std::bad_alloc during cross-distro
# discovery/deserialization.  Depth remains Fast DDS on isolated domain 43.
export RMW_IMPLEMENTATION="${G1_ARM_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
GB10_ROS_DISTRO="${GB10_ROS_DISTRO:-$([[ -r /opt/ros/jazzy/setup.bash ]] && echo jazzy || echo humble)}"
g1_source_ros "${ROOT_DIR}" "${GB10_ROS_DISTRO}"
if ! ros2 pkg prefix "${RMW_IMPLEMENTATION}" >/dev/null 2>&1; then
  echo "Missing ros-${GB10_ROS_DISTRO}-${RMW_IMPLEMENTATION#rmw_} on the GB10." >&2
  g1_console_error "Missing ${RMW_IMPLEMENTATION}. See ${G1_ACTIVE_LOG_FILE}"
  exit 1
fi
if [[ "${ROS_INTERFACE}" == "auto" ]]; then
  ROS_INTERFACE="$(ip -4 route get "${ROBOT_HOST}" | awk 'NR==1 {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')"
fi
if [[ -z "${ROS_INTERFACE}" ]]; then
  echo "Could not determine the GB10 interface used to reach ${ROBOT_HOST}." >&2
  g1_console_error "Could not determine the interface to ${ROBOT_HOST}. See ${G1_ACTIVE_LOG_FILE}"
  exit 1
fi
g1_configure_cyclonedds manual-arm-gb10 "${ROS_INTERFACE}" "${ROBOT_HOST}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

client_args=("$@")
if [[ "${command_name}" == "move" || "${command_name}" == "ik" || "${command_name}" == "point" ]]; then
  trace_requested=0
  for value in "${client_args[@]}"; do
    if [[ "${value}" == "--trace-output" ]]; then
      trace_requested=1
      break
    fi
  done
  if [[ "${trace_requested}" == "0" ]]; then
    automatic_trace="${G1_ACTIVE_LOG_FILE%.log}.trace.log"
    client_args+=(--trace-output "${automatic_trace}")
    ln -sfn "$(basename "${automatic_trace}")" \
      "$(dirname "${automatic_trace}")/latest-trace.log"
  fi
fi
g1_log_command "$0" "${client_args[@]}"
echo "ROS transport: RMW=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} interface=${ROS_INTERFACE} robot=${ROBOT_HOST}"
echo "Git revision: $(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

set +e
"${GB10_PYTHON}" -m object_tracking.manual_arm_cli "${client_args[@]}"
status=$?
set -e
summary="$("${GB10_PYTHON}" - "${result_file}" "${command_name}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
command = sys.argv[2]
if not path.is_file():
    print(f"Arm {command} produced no result record")
    raise SystemExit
try:
    result = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(f"Arm {command} result record is unreadable")
    raise SystemExit
if not result.get("ok", False):
    print(str(result.get("error") or f"arm {command} failed"))
    raise SystemExit
bridge = result.get("bridge") if isinstance(result.get("bridge"), dict) else result
if command == "inspect":
    print(
        "Arm bridge is "
        f"{bridge.get('state', 'unknown')}; mode={bridge.get('motion_mode_name')}; "
        f"standing={bridge.get('standing')}; motors_healthy={bridge.get('motor_state_healthy')}"
    )
elif command in {"move", "ik", "point"}:
    details = [f"steps={result.get('guarded_steps', 0)}"]
    if result.get("point_route_kind"):
        details.append(f"route={result['point_route_kind']}")
    if result.get("point_stages"):
        details.append(f"stages={result['point_stages']}")
    if result.get("ik_position_error_m") is not None:
        details.append(f"position_error={100 * float(result['ik_position_error_m']):.1f}cm")
    if result.get("point_angular_error_deg") is not None:
        details.append(f"ray_error={float(result['point_angular_error_deg']):.1f}deg")
    details.append(f"returned={result.get('returned')}")
    print(f"Arm {command} completed (" + ", ".join(details) + ")")
else:
    print(f"Arm {command} completed; state={bridge.get('state', 'unknown')}")
PY
)"
if [[ "${status}" == "0" ]]; then
  g1_console "${summary}"
else
  g1_console_error "${summary}"
fi
g1_console "Details: ${G1_ACTIVE_LOG_FILE}"
summary_printed=1
exit "${status}"
