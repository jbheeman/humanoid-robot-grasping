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
  # Robot-local native Unitree SDK2 already owns a CycloneDDS domain.  The
  # robot's project ROS node therefore uses Fast DDS by default; GB10 keeps
  # CycloneDDS unless its launcher explicitly chooses otherwise.
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
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
  local legacy_peers_xml=""
  local -a peers=()
  local -a interfaces=()

  export ROS_DOMAIN_ID="${domain_id}"
  if [[ "${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" != "rmw_cyclonedds_cpp" ]]; then
    # Native Unitree SDK2 owns CycloneDDS on the robot.  Pin Fast DDS project
    # traffic to the first requested project NIC (normally wlan0), otherwise
    # it advertises the private eth0 control-network address and GB10 cannot
    # discover the bridge.
    local fastdds_interface="${interface_name%%,*}"
    fastdds_interface="${fastdds_interface//[[:space:]]/}"
    if [[ "${fastdds_interface}" == "auto" ]]; then
      local route_peer="${peers_csv%%,*}"
      route_peer="${route_peer//[[:space:]]/}"
      fastdds_interface="$(
        ip -4 route get "${route_peer}" 2>/dev/null |
          awk 'NR==1 {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}' || true
      )"
    fi
    if [[ -z "${fastdds_interface}" ]]; then
      echo "Could not determine the Fast DDS interface used to reach ${peers_csv}." >&2
      return 1
    fi
    local fastdds_ip
    fastdds_ip="$(
      ip -4 -o addr show dev "${fastdds_interface}" 2>/dev/null |
        awk 'NR==1 {split($4, a, "/"); print a[1]}' || true
    )"
    if [[ -z "${fastdds_ip}" ]]; then
      echo "Could not determine an IPv4 address for Fast DDS interface ${fastdds_interface}." >&2
      return 1
    fi
    local fastdds_config="${runtime_root}/fastdds-${role}.xml"
    mkdir -p "${runtime_root}"
    umask 077
    cat > "${fastdds_config}" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>g1_project_udp</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList><address>${fastdds_ip}</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="g1_project" is_default_profile="true">
    <rtps>
      <userTransports><transport_id>g1_project_udp</transport_id></userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
EOF
    export FASTRTPS_DEFAULT_PROFILES_FILE="${fastdds_config}"
    export FASTDDS_DEFAULT_PROFILES_FILE="${fastdds_config}"
    unset CYCLONEDDS_URI G1_CYCLONEDDS_CONFIG
    return 0
  fi

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
    legacy_peers_xml+="<Peer address=\"${peer}\" />"
  done
  if [[ -z "${peers_xml}" ]]; then
    echo "At least one CycloneDDS static peer is required." >&2
    return 2
  fi

  mkdir -p "${runtime_root}"
  umask 077
  if [[ "${ROS_DISTRO:-}" == "foxy" ]]; then
    local legacy_interface="${interface_name%%,*}"
    legacy_interface="${legacy_interface//[[:space:]]/}"
    printf '%s\n' \
      '<?xml version="1.0" encoding="UTF-8"?>' \
      '<CycloneDDS>' \
      '  <Domain id="any">' \
      '    <General>' \
      "      <NetworkInterfaceAddress>${legacy_interface}</NetworkInterfaceAddress>" \
      '      <AllowMulticast>false</AllowMulticast>' \
      '    </General>' \
      '    <Discovery>' \
      '      <ParticipantIndex>auto</ParticipantIndex>' \
      '      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>' \
      "      <Peers>${legacy_peers_xml}</Peers>" \
      '    </Discovery>' \
      '  </Domain>' \
      '</CycloneDDS>' > "${config_file}"
  else
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
  fi

  export CYCLONEDDS_URI="${config_file}"
  export G1_CYCLONEDDS_CONFIG="${config_file}"
}
