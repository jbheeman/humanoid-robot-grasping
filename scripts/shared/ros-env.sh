#!/usr/bin/env bash

# Shared ROS 2 environment helpers for robot and GB10 launchers.
# This file is sourced; callers should enable their own strict shell options.

g1_source_ros() {
  local root_dir="$1"
  local expected_distro="$2"
  local ros_setup="/opt/ros/${expected_distro}/setup.bash"
  local unitree_setup="${root_dir}/.ros/${expected_distro}/unitree/setup.bash"
  local project_setup="${root_dir}/.ros/${expected_distro}/project/setup.bash"

  if [[ ! -r "${ros_setup}" ]]; then
    echo "ROS 2 ${expected_distro} is required: missing ${ros_setup}" >&2
    return 1
  fi
  # ROS setup scripts may inspect unset variables, so temporarily relax nounset.
  set +u
  source "${ros_setup}"
  set -u
  if [[ "${ROS_DISTRO:-}" != "${expected_distro}" ]]; then
    echo "Expected ROS_DISTRO=${expected_distro}, got ${ROS_DISTRO:-unset}." >&2
    return 1
  fi
  if [[ ! -r "${unitree_setup}" || ! -r "${project_setup}" ]]; then
    echo "ROS workspaces are not built for ${expected_distro}." >&2
    echo "Run: uv run g1 setup $([[ "${expected_distro}" == "foxy" ]] && echo robot || echo gb10)" >&2
    return 1
  fi
  set +u
  source "${unitree_setup}"
  source "${project_setup}"
  set -u
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
}

g1_configure_cyclonedds() {
  local role="$1"
  local interface_name="$2"
  local peers_csv="$3"
  local domain_id="${4:-0}"
  local runtime_root="${XDG_RUNTIME_DIR:-/tmp}/g1-ros-${UID}"
  local config_file="${runtime_root}/cyclonedds-${role}.xml"
  local interface_xml=""
  local interface
  local peer
  local peers_xml=""
  local -a peers=()
  local -a interfaces=()

  if [[ ! "${role}" =~ ^[a-z0-9_-]+$ ]]; then
    echo "Invalid ROS role: ${role}" >&2
    return 2
  fi
  if [[ ! "${domain_id}" =~ ^[0-9]+$ ]] || (( domain_id > 232 )); then
    echo "ROS_DOMAIN_ID must be an integer from 0 to 232." >&2
    return 2
  fi
  if [[ "${interface_name}" == "auto" ]]; then
    interface_xml='<NetworkInterface autodetermine="true" priority="default" />'
  else
    IFS=',' read -r -a interfaces <<< "${interface_name}"
    for interface in "${interfaces[@]}"; do
      interface="${interface//[[:space:]]/}"
      if [[ ! "${interface}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
        echo "Invalid ROS interface name: ${interface}" >&2
        return 2
      fi
      interface_xml+="<NetworkInterface name=\"${interface}\" priority=\"default\" />"
    done
    if [[ -z "${interface_xml}" ]]; then
      echo "At least one ROS interface is required." >&2
      return 2
    fi
  fi

  IFS=',' read -r -a peers <<< "${peers_csv}"
  for peer in "${peers[@]}"; do
    peer="${peer//[[:space:]]/}"
    [[ -z "${peer}" ]] && continue
    if [[ ! "${peer}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
      echo "Invalid CycloneDDS peer address: ${peer}" >&2
      return 2
    fi
    peers_xml+="<Peer Address=\"${peer}\" />"
  done
  if [[ -z "${peers_xml}" ]]; then
    echo "At least one CycloneDDS static peer is required." >&2
    return 2
  fi

  mkdir -p "${runtime_root}"
  umask 077
  printf '%s\n' \
    '<?xml version="1.0" encoding="UTF-8"?>' \
    '<CycloneDDS>' \
    '  <Domain Id="any">' \
    '    <General>' \
    "      <Interfaces>${interface_xml}</Interfaces>" \
    '      <AllowMulticast>false</AllowMulticast>' \
    '    </General>' \
    '    <Discovery>' \
    '      <ParticipantIndex>auto</ParticipantIndex>' \
    '      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>' \
    "      <Peers>${peers_xml}</Peers>" \
    '    </Discovery>' \
    '  </Domain>' \
    '</CycloneDDS>' > "${config_file}"

  export ROS_DOMAIN_ID="${domain_id}"
  export CYCLONEDDS_URI="${config_file}"
  export G1_CYCLONEDDS_CONFIG="${config_file}"
}
