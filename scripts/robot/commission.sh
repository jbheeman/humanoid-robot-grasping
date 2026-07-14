#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_PYTHON="${ROBOT_PYTHON:-${ROOT_DIR}/robot/.venv/bin/python}"
CLIENT_IP="${CLIENT_IP:-${GB10_HOST:-192.168.0.66}}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-wlan0}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"
ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-42}"
HARDWARE_DOMAIN_ID="${HARDWARE_DOMAIN_ID:-0}"
ALLOW_MOVEMENT="${ALLOW_MOVEMENT:-0}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-}"
G1_ROBOT_ID="${G1_ROBOT_ID:-}"
CALIBRATION="${CALIBRATION:-}"
COMMISSIONING_ROOT="${COMMISSIONING_ROOT:-${ROOT_DIR}/runs/research/arm_commissioning}"
COMMISSIONING_PROFILE="${COMMISSIONING_PROFILE:-${HOME}/.config/g1-grasping/right-arm-home.json}"
DEPTH_SOURCE="${DEPTH_SOURCE:-auto}"
RGB_MODE="${RGB_MODE:-unitree}"
# Keep rclpy separate from the native SDK2 CycloneDDS domain in this process.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

if [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "Missing robot environment: ${ROBOT_PYTHON}" >&2
  echo "Run uv run g1 setup robot first." >&2
  exit 1
fi
if [[ -z "${CLIENT_IP}" ]]; then
  echo "CLIENT_IP (the GB10 address) is required." >&2
  exit 1
fi
if [[ -z "${G1_ROBOT_ID}" ]]; then
  echo "G1_ROBOT_ID is required so saved poses remain robot-specific." >&2
  exit 1
fi
if [[ "${ALLOW_MOVEMENT}" == "1" && -z "${EXPECTED_MOTION_MODE}" ]]; then
  echo "EXPECTED_MOTION_MODE is required for movement." >&2
  exit 1
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
g1_source_ros "${ROOT_DIR}" foxy
g1_configure_cyclonedds \
  robot-commissioning "${ROBOT_INTERFACE}" \
  "${CLIENT_IP},${UNITREE_CONTROL_PEER}" "${ROS_DOMAIN_ID}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
# The stock robot image keeps SDK2 as a source checkout.  This is deliberately
# separate from ROS: only the local hardware adapter imports it.
UNITREE_SDK_PYTHONPATH="${UNITREE_SDK_PYTHONPATH:-${HOME}/unitree_sdk2_python}"
if [[ ! -f "${UNITREE_SDK_PYTHONPATH}/unitree_sdk2py/__init__.py" ]]; then
  echo "Native Unitree SDK2 was not found at ${UNITREE_SDK_PYTHONPATH}." >&2
  echo "Set UNITREE_SDK_PYTHONPATH to the unitree_sdk2_python checkout." >&2
  exit 1
fi
export PYTHONPATH="${UNITREE_SDK_PYTHONPATH}:${PYTHONPATH}"
# Sourcing ROS Foxy places its older CycloneDDS library ahead of the SDK's
# Python extension.  Project ROS uses Fast DDS in this process, while SDK2
# needs the native Unitree-compatible libddsc for motor-state traffic.
UNITREE_SDK_DDS_LIBRARY_DIR="${UNITREE_SDK_DDS_LIBRARY_DIR:-/usr/local/lib}"
if [[ ! -f "${UNITREE_SDK_DDS_LIBRARY_DIR}/libddsc.so.0" ]]; then
  echo "Native Unitree CycloneDDS library not found: ${UNITREE_SDK_DDS_LIBRARY_DIR}/libddsc.so.0" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${UNITREE_SDK_DDS_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

args=(
  --control-mode commissioning
  --robot-id "${G1_ROBOT_ID}"
  --commissioning-root "${COMMISSIONING_ROOT}"
  --commissioning-profile "${COMMISSIONING_PROFILE}"
  --depth-source "${DEPTH_SOURCE}"
  --disable-depth
  --hardware-interface "${HARDWARE_INTERFACE}"
  --hardware-domain-id "${HARDWARE_DOMAIN_ID}"
)
if [[ -n "${CALIBRATION}" ]]; then
  args+=(--calibration "${CALIBRATION}")
fi
if [[ "${ALLOW_MOVEMENT}" == "1" ]]; then
  args+=(--allow-movement --expected-motion-mode "${EXPECTED_MOTION_MODE}")
fi

echo
echo "============================================================"
echo " G1 RIGHT-ARM COMMISSIONING (ROS 2)"
echo "============================================================"
echo "Movement permitted: ${ALLOW_MOVEMENT}"
echo "Robot ID:           ${G1_ROBOT_ID}"
echo "Expected mode:       ${EXPECTED_MOTION_MODE:-unconfigured}"
echo "ROS interface:       ${ROBOT_INTERFACE}"
echo "Motor DDS interface: ${HARDWARE_INTERFACE}"
echo "ROS domain:          ${ROS_DOMAIN_ID}"
echo "Motor DDS domain:    ${HARDWARE_DOMAIN_ID}"
echo "Research data:       ${COMMISSIONING_ROOT}"
echo "Promoted home:       ${COMMISSIONING_PROFILE}"
echo
echo "With the GB10 server running, open:"
echo "  http://${CLIENT_IP}:8000/commissioning/"
echo "For off-LAN access, tunnel only GB10 port 8000."
echo
echo "The webpage Stop button is not a physical e-stop."
echo "============================================================"
echo

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The same GB10 process serves vision at / and commissioning at
# /commissioning/.  Start the RGB relay here so commissioning never silently
# leaves the vision page without camera packets.
case "${RGB_MODE}" in
  unitree)
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
      "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  30fps)
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" "${ROOT_DIR}/scripts/robot/rgb-30fps.sh" &
    pids+=("$!")
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
      "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  highfps-service)
    # The root-owned boot service owns /dev/video4 and emits the same RTP
    # multicast.  Commissioning only needs the relay to GB10.
    ROBOT_INTERFACE="${HARDWARE_INTERFACE}" CLIENT_IP="${CLIENT_IP}" \
      "${ROOT_DIR}/scripts/robot/rgb-relay.sh" &
    pids+=("$!")
    ;;
  off)
    echo "RGB relay disabled (RGB_MODE=off)."
    ;;
  *)
    echo "RGB_MODE must be unitree, 30fps, highfps-service, or off; got: ${RGB_MODE}" >&2
    exit 2
    ;;
esac

"${ROBOT_PYTHON}" "${ROOT_DIR}/scripts/robot/ros_node.py" "${args[@]}" &
pids+=("$!")
wait -n "${pids[@]}"
echo "A robot process exited; stopping the process group." >&2
exit 1
