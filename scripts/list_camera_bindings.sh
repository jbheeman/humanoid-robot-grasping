#!/usr/bin/env bash
set -euo pipefail

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
    }'
else
  echo "lsof missing: install lsof for a one-shot owner map"
fi

echo
echo "Tip: when streams fail, compare process bindings:"
echo "  ps -eo pid,user,cmd | grep -E 'videohub_pc4|run_yolo_stream|yolo_stream_server|gst-launch-1.0' | grep -v grep"
