#!/usr/bin/env bash
set -euo pipefail

PROCESS_NAME="${1:-}"

if [[ -n "${PROCESS_NAME}" ]]; then
  echo "Video devices opened by process '${PROCESS_NAME}':"
  found_process=0
  found_device=0
  found_configured_device=0
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    found_process=1
    echo "  PID ${pid}: $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    while read -r configured_device; do
      [[ -n "${configured_device}" ]] || continue
      found_configured_device=1
      if [[ -e "${configured_device}" ]]; then
        echo "    configured argument -> ${configured_device} (exists)"
      else
        echo "    configured argument -> ${configured_device} (MISSING)"
      fi
    done < <(tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null | grep -E '^/dev/video[0-9]+$' || true)
    for fd in "/proc/${pid}/fd/"*; do
      [[ -e "${fd}" ]] || continue
      target="$(readlink -f "${fd}" 2>/dev/null || true)"
      if [[ "${target}" =~ ^/dev/video[0-9]+$ ]]; then
        echo "    fd $(basename "${fd}") -> ${target}"
        found_device=1
      fi
    done
  done < <(pgrep -x "${PROCESS_NAME}" 2>/dev/null || true)

  if [[ "${found_process}" == "0" ]]; then
    echo "  No exact process named '${PROCESS_NAME}' is running."
  elif [[ "${found_device}" == "0" ]]; then
    echo "  No open /dev/video* file descriptor found."
    if [[ "${EUID}" != "0" ]]; then
      echo "  Retry with sudo to rule out /proc permission restrictions:"
      echo "    sudo ./scripts/list_camera_bindings.sh ${PROCESS_NAME}"
    elif [[ "${found_configured_device}" == "1" ]]; then
      echo "  The process is not currently using its configured video device."
    fi
  fi
  echo
fi

echo "Detected /dev/video devices (sysfs):"
if [[ -d /sys/class/video4linux ]]; then
  for dev in /sys/class/video4linux/video*; do
    [[ -e "$dev" ]] || continue
    name="$(cat "${dev}/name" 2>/dev/null || echo unknown)"
    devnode="/dev/$(basename "$dev")"
    if [[ -e "$devnode" ]]; then
      echo "  ${devnode}: ${name}"
    fi
  done
else
  echo "  /sys/class/video4linux is unavailable"
fi

echo
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo "v4l2-ctl --list-devices:"
  v4l2-ctl --list-devices || true
else
  echo "v4l2-ctl missing: install v4l-utils to get richer device labels"
fi

echo
if command -v fuser >/dev/null 2>&1; then
  echo "Processes holding /dev/video* (fuser):"
  for dev in /dev/video*; do
    [[ -e "${dev}" ]] || continue
    output="$(fuser -v "${dev}" 2>/dev/null || true)"
    if [[ -n "${output}" ]]; then
      echo "  ${dev}"
      echo "${output}" | sed "s/^/    /"
    fi
  done
else
  echo "fuser missing: install psmisc to show device owners"
fi

echo
if command -v lsof >/dev/null 2>&1; then
  echo "Open file handles on video devices (lsof):"
  lsof +w -Fcn -- /dev/video* 2>/dev/null | awk '
    BEGIN { RS = "\0"; ORS = "\n" }
    /n\/dev\/video/ {
      cmd=""; pid=""; file="";
      split($0,a,"\n");
      for (i in a) {
        entry=a[i];
        if (entry ~ /^p/) { pid=substr(entry,2); }
        else if (entry ~ /^c/) { cmd=substr(entry,2); }
        else if (entry ~ /^n/) { file=substr(entry,2); }
      }
      if (pid != "" && cmd != "" && file != "") {
        print "  " cmd " (" pid ") -> " file;
      }
    }' || true
else
  echo "lsof missing: install lsof for a one-shot owner map"
fi

echo
echo "Tip: when streams fail, compare process bindings:"
echo "  ps -eo pid,user,cmd | grep -E 'videohub_pc4|run_yolo_stream|yolo_stream_server|gst-launch-1.0' | grep -v grep"
echo "  sudo ./scripts/list_camera_bindings.sh videohub_pc4_ch"
