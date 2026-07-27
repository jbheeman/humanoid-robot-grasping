#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
PROJECT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
DEPTH_ROS_DOMAIN_ID="${G1_DEPTH_ROS_DOMAIN_ID:-43}"

source "${ROOT_DIR}/scripts/shared/run-logging.sh"
g1_begin_run_log "${ROOT_DIR}" "robot-vision-pointing"
g1_log_command "$0" "$@"

if ! systemctl is-active --quiet g1-highfps-camera.service; then
  echo "The native 60 FPS camera publisher is not running." >&2
  echo "Run: sudo systemctl enable --now g1-highfps-camera.service" >&2
  g1_console_error "The 60 FPS camera service is not running. See ${G1_ACTIVE_LOG_FILE}"
  exit 1
fi
if pgrep -f "${ROOT_DIR}/scripts/robot/ros_node.py" >/dev/null 2>&1; then
  echo "Another project robot node is already running." >&2
  echo "Stop the existing robot/manual-arm launcher with Ctrl+C, then retry." >&2
  g1_console_error "Another robot bridge is already running. See ${G1_ACTIVE_LOG_FILE}"
  exit 1
fi

pids=()
critical_pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
  "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
pids+=("$!")
critical_pids+=("$!")

(
  # Use the same Fast DDS transport proven by the remote manual-arm bridge.
  # Foxy CycloneDDS intermittently fails to deserialize the large Jazzy depth
  # envelope after discovery, even while its local publisher is healthy.
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  source "${ROOT_DIR}/scripts/shared/ros-env.sh"
  g1_source_ros "${ROOT_DIR}" foxy
  g1_configure_cyclonedds depth-robot \
    "${PROJECT_INTERFACE}" "${CLIENT_IP}" "${DEPTH_ROS_DOMAIN_ID}"
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  depth_args=(
    --depth-only
    --depth-source librealsense
    --depth-width "${DEPTH_WIDTH:-848}"
    --depth-height "${DEPTH_HEIGHT:-480}"
    --depth-capture-fps "${DEPTH_CAPTURE_FPS:-60}"
    --depth-publish-fps "${DEPTH_PUBLISH_FPS:-30}"
  )
  if [[ -n "${DEPTH_SERIAL:-}" ]]; then
    depth_args+=(--depth-serial "${DEPTH_SERIAL}")
  fi
  exec "${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${depth_args[@]}"
) &
pids+=("$!")
critical_pids+=("$!")

MANUAL_ARM_PROFILE="${MANUAL_ARM_PROFILE:-xr}" \
CLIENT_IP="${CLIENT_IP}" \
ROBOT_INTERFACE="${PROJECT_INTERFACE}" \
HARDWARE_INTERFACE="${HARDWARE_INTERFACE}" \
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER}" \
G1_PROJECT_ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  "${ROOT_DIR}/scripts/robot/manual-arm.sh" move &
manual_arm_pid="$!"
pids+=("${manual_arm_pid}")

echo "Vision pointing stack started initially DISARMED."
echo "  RGB: native 960x540@60 relay to ${CLIENT_IP}:5600"
echo "  Depth: isolated Fast DDS /g1/depth on domain ${DEPTH_ROS_DOMAIN_ID}"
echo "  Arm: isolated Fast DDS XR manual bridge on domain ${ROS_DOMAIN_ID} (non-critical)"
echo "  Combined log: ${G1_ACTIVE_LOG_FILE}"
g1_console "Vision pointing stack starting DISARMED: RGB 60 FPS, depth domain ${DEPTH_ROS_DOMAIN_ID}, arm domain ${ROS_DOMAIN_ID}."

# Bash 5.0 can wake `wait -n PID...` for a different child than those listed.
# Poll the two explicitly critical PIDs instead so a manual-arm DDS abort can
# never be mistaken for an RGB/depth failure and tear down healthy vision.
monitor_critical_processes() {
  local arm_exit_reported=0
  local pid
  local process_state
  while true; do
    if (( arm_exit_reported == 0 )); then
      process_state="$(ps -o stat= -p "${manual_arm_pid}" 2>/dev/null || true)"
      if [[ -z "${process_state}" || "${process_state}" == Z* ]]; then
        set +e
        wait "${manual_arm_pid}"
        local arm_status=$?
        set -e
        echo "Manual arm bridge exited (code ${arm_status}); RGB/depth remain active." >&2
        echo "After diagnosing it, the operator may restart only the disarmed bridge with:" >&2
        echo "  G1_PROJECT_ROS_DOMAIN_ID=${ROS_DOMAIN_ID} scripts/robot/manual-arm.sh move" >&2
        arm_exit_reported=1
      fi
    fi
    for pid in "${critical_pids[@]}"; do
      process_state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
      if [[ -z "${process_state}" || "${process_state}" == Z* ]]; then
        set +e
        wait "${pid}"
        status=$?
        set -e
        return "${status}"
      fi
    done
    sleep 0.20
  done
}

set +e
monitor_critical_processes
status=$?
set -e
echo "A critical vision-pointing process exited; stopping the stack." >&2
if [[ "${status}" == "0" || "${status}" == "130" ]]; then
  g1_console "Vision pointing stack stopped."
else
  g1_console_error "A vision-pointing process exited (code ${status}). Details: ${G1_ACTIVE_LOG_FILE}"
fi
exit "${status:-1}"
