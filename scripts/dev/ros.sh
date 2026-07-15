#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROLE=""
PEER=""
INTERFACE="auto"
DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
PROFILE="full"
UNITREE_CONTROL_PEER="${UNITREE_CONTROL_PEER:-192.168.123.1}"

usage() {
  echo "Usage: g1 inspect ros --role {robot|gb10} --peer IP [--interface NAME] [--domain-id ID] [--profile {full|manual}]"
  echo "Lists ROS nodes/topics and verifies the project and Unitree interface types without commanding movement."
}

while (( $# > 0 )); do
  case "$1" in
    --role)
      ROLE="${2:-}"
      shift 2
      ;;
    --peer)
      PEER="${2:-}"
      shift 2
      ;;
    --interface)
      INTERFACE="${2:-}"
      shift 2
      ;;
    --domain-id)
      DOMAIN_ID="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ROLE}" != "robot" && "${ROLE}" != "gb10" ]]; then
  echo "--role must be robot or gb10." >&2
  usage >&2
  exit 2
fi
if [[ "${PROFILE}" != "full" && "${PROFILE}" != "manual" ]]; then
  echo "--profile must be full or manual." >&2
  exit 2
fi
if [[ -z "${PEER}" ]]; then
  echo "--peer is required (GB10 IP on robot; robot IP on GB10)." >&2
  exit 2
fi

source "${ROOT_DIR}/scripts/shared/ros-env.sh"
if [[ "${ROLE}" == "robot" ]]; then
  g1_source_ros "${ROOT_DIR}" foxy
  g1_configure_cyclonedds inspect-robot "${INTERFACE}" \
    "${PEER},${UNITREE_CONTROL_PEER}" "${DOMAIN_ID}"
else
  g1_source_ros "${ROOT_DIR}" jazzy
  g1_configure_cyclonedds inspect-gb10 "${INTERFACE}" "${PEER}" "${DOMAIN_ID}"
fi

echo "ROS_DISTRO=${ROS_DISTRO} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo
echo "Nodes:"
timeout 8 ros2 node list || true
echo
echo "Topics:"
topics="$(timeout 8 ros2 topic list || true)"
printf '%s\n' "${topics}"

if [[ "${PROFILE}" == "manual" ]]; then
  expected_topics=(
    /g1/arm/state
    /g1/arm_control/left/command
    /g1/arm_control/right/command
    /g1/arm_control/heartbeat
    /g1/arm_control/status
    /g1/arm_control/joint_states
    /g1/arm_control/request
    /g1/arm_control/response
  )
else
  expected_topics=(
    /g1/arm/state
    /g1/depth
    /g1/commissioning/state
  )
fi
echo
echo "Expected interface check:"
missing=0
for topic in "${expected_topics[@]}"; do
  if grep -Fxq "${topic}" <<< "${topics}"; then
    topic_type="$(timeout 5 ros2 topic type "${topic}" 2>/dev/null || true)"
    echo "  ok      ${topic} ${topic_type}"
  else
    echo "  missing ${topic}"
    missing=1
  fi
done

if [[ "${missing}" == "1" ]]; then
  echo "One or more expected topics is absent; verify both launchers, peers, interface, and domain." >&2
  exit 1
fi
echo "Read-only ROS 2 discovery check passed."
