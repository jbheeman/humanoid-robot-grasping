#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
CALIBRATION="${CALIBRATION:-${ROOT_DIR}/runs/localization/g1-tabletop-calibration.json}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-ai}"
MAX_TILT_DEG="${MAX_TILT_DEG:-8}"
MAX_WAIST_DEVIATION_DEG="${MAX_WAIST_DEVIATION_DEG:-12}"
ARM_WEIGHT_RAMP_S="${ARM_WEIGHT_RAMP_S:-0.75}"
ARM_MAX_VELOCITY_RAD_S="${ARM_MAX_VELOCITY_RAD_S:-0.30}"
ARM_MAX_ACCELERATION_RAD_S2="${ARM_MAX_ACCELERATION_RAD_S2:-0.75}"
ARM_MAX_JERK_RAD_S3="${ARM_MAX_JERK_RAD_S3:-4.0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-90}"
READY_STABLE_SAMPLES="${READY_STABLE_SAMPLES:-4}"
# RGB uses the root-owned 960x540@60 GStreamer service. Librealsense opens only
# the depth interface so NVENC and ROS depth serialization cannot block the
# same capture loop.
DEPTH_CAPTURE_FPS="${DEPTH_CAPTURE_FPS:-60}"
# Domain 42 is also used by older project processes and has repeatedly left
# the live G1 and GB10 participants undiscovered.  The live bunny test uses a
# dedicated domain that is verified end-to-end during commissioning.
PROJECT_ROS_DOMAIN_ID="${G1_PROJECT_ROS_DOMAIN_ID:-43}"
LOG_DIR="${ROOT_DIR}/logs/g1-bunny-test"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "G1 bunny-test log: ${LOG_FILE}"

while (($#)); do
  case "$1" in
    --client-ip)
      CLIENT_IP="${2:?--client-ip requires an address}"
      shift 2
      ;;
    --calibration)
      CALIBRATION="${2:?--calibration requires a path}"
      shift 2
      ;;
    --expected-motion-mode)
      EXPECTED_MOTION_MODE="${2:?--expected-motion-mode requires a mode}"
      shift 2
      ;;
    --max-tilt-deg)
      MAX_TILT_DEG="${2:?--max-tilt-deg requires a number}"
      shift 2
      ;;
    --max-waist-deviation-deg)
      MAX_WAIST_DEVIATION_DEG="${2:?--max-waist-deviation-deg requires a number}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: uv run g1 robot bunny-test [--client-ip 192.168.0.66]

Starts the foreground robot bridge with movement permission, waits until the
GB10 has fresh RGB/depth, bunny detection, table geometry, and safe IK, then
automatically enables one guarded tracking session. Ctrl-C stops the arm and
all robot-local processes.
EOF
      exit 0
      ;;
    *)
      echo "Unknown bunny-test option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CALIBRATION}" ]]; then
  echo "Missing calibration: ${CALIBRATION}" >&2
  exit 1
fi

if ! systemctl is-active --quiet g1-highfps-camera.service; then
  echo "Starting the 960x540@60 GStreamer camera service (sudo may prompt)..."
  sudo systemctl start g1-highfps-camera.service
fi
if ! systemctl is-active --quiet g1-highfps-camera.service; then
  echo "g1-highfps-camera.service did not become active." >&2
  sudo systemctl status g1-highfps-camera.service --no-pager >&2 || true
  exit 1
fi

bridge_pid=""
bridge_pgid=""
session_enabled=0
cleaning_up=0
stop_remote_arm() {
  if [[ "${session_enabled}" == "1" ]]; then
    python3 - "${CLIENT_IP}" <<'PY' >/dev/null 2>&1 || true
import json
import sys
import urllib.request

request = urllib.request.Request(
    f"http://{sys.argv[1]}:8000/api/v1/arm/stop",
    data=json.dumps({"reason": "operator_bunny_test_stopped"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
urllib.request.urlopen(request, timeout=2).read()
PY
  fi
}
cleanup() {
  if [[ "${cleaning_up}" == "1" ]]; then
    return
  fi
  cleaning_up=1
  stop_remote_arm
  if [[ -n "${bridge_pgid}" ]]; then
    kill -TERM -- "-${bridge_pgid}" 2>/dev/null || true
    for _ in {1..50}; do
      if ! kill -0 -- "-${bridge_pgid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 -- "-${bridge_pgid}" 2>/dev/null; then
      kill -KILL -- "-${bridge_pgid}" 2>/dev/null || true
    fi
  elif [[ -n "${bridge_pid}" ]]; then
    kill -TERM "${bridge_pid}" 2>/dev/null || true
  fi
  if [[ -n "${bridge_pid}" ]]; then
    wait "${bridge_pid}" 2>/dev/null || true
  fi
  # Librealsense can retain the V4L2 interface briefly after process exit.
  sleep 1
}
on_signal() {
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

echo "Starting guarded live bunny test"
echo "  GB10:       ${CLIENT_IP}"
echo "  calibration: ${CALIBRATION}"
echo "  motion mode: ${EXPECTED_MOTION_MODE}"
echo "  tilt limit:  ${MAX_TILT_DEG} degrees"
echo "  waist limit: ${MAX_WAIST_DEVIATION_DEG} degrees"
echo "  arm limits:  ${ARM_MAX_VELOCITY_RAD_S} rad/s, ${ARM_MAX_ACCELERATION_RAD_S2} rad/s^2, ${ARM_MAX_JERK_RAD_S3} rad/s^3"
echo "  weight ramp: ${ARM_WEIGHT_RAMP_S} s"
echo "  depth rate:  ${DEPTH_CAPTURE_FPS} Hz"
echo "  ROS domain:  ${PROJECT_ROS_DOMAIN_ID}"
echo "Keep the physical E-stop in hand. Ctrl-C performs a controlled stop."

CLIENT_IP="${CLIENT_IP}" \
CALIBRATION="${CALIBRATION}" \
ALLOW_MOVEMENT=1 \
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE}" \
MAX_TILT_DEG="${MAX_TILT_DEG}" \
MAX_WAIST_DEVIATION_DEG="${MAX_WAIST_DEVIATION_DEG}" \
ARM_WEIGHT_RAMP_S="${ARM_WEIGHT_RAMP_S}" \
ARM_MAX_VELOCITY_RAD_S="${ARM_MAX_VELOCITY_RAD_S}" \
ARM_MAX_ACCELERATION_RAD_S2="${ARM_MAX_ACCELERATION_RAD_S2}" \
ARM_MAX_JERK_RAD_S3="${ARM_MAX_JERK_RAD_S3}" \
DEPTH_CAPTURE_FPS="${DEPTH_CAPTURE_FPS}" \
RGB_MODE=highfps-service \
G1_PROJECT_ROS_DOMAIN_ID="${PROJECT_ROS_DOMAIN_ID}" \
QUIET_HEALTHY_DEPTH=1 \
  setsid "${ROOT_DIR}/scripts/robot/start.sh" &
bridge_pid="$!"
bridge_pgid="$(
  ps -o pgid= -p "${bridge_pid}" 2>/dev/null |
    tr -d '[:space:]'
)"
if [[ -z "${bridge_pgid}" ]]; then
  echo "Could not determine the robot process group." >&2
  exit 1
fi

python3 - "${CLIENT_IP}" "${READY_TIMEOUT_S}" "${READY_STABLE_SAMPLES}" <<'PY'
import json
import sys
import time
import urllib.request

host = sys.argv[1]
timeout_s = float(sys.argv[2])
stable_samples = int(sys.argv[3])
if stable_samples < 1:
    raise SystemExit("READY_STABLE_SAMPLES must be positive")
url = f"http://{host}:8000/health"
deadline = time.monotonic() + timeout_s
last_reason = "GB10 has not responded"
last_reported_reason = None
last_report_at = 0.0
ready_samples = 0
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            health = json.load(response)
        tracking = health.get("arm_tracking") or {}
        command = tracking.get("predicted_bounded_arm_command_rad")
        ready = (
            tracking.get("mode") == "execute"
            and tracking.get("arm_state") == "DISARMED"
            and tracking.get("ik_status") == "ok"
            and isinstance(command, list)
            and len(command) == 7
            and tracking.get("reason") == "arm_not_explicitly_enabled"
            and float(tracking.get("depth_age_ms", 1e9)) <= 200.0
            and float(tracking.get("processing_latency_ms", 1e9)) <= 250.0
        )
        ready_samples = ready_samples + 1 if ready else 0
        if ready_samples >= stable_samples:
            print(
                "GB10 ready: "
                f"confidence={tracking.get('detector_confidence')} "
                f"IK={tracking.get('ik_step_type')} "
                f"latency_ms={tracking.get('processing_latency_ms')} "
                f"stable_samples={ready_samples}",
                flush=True,
            )
            break
        last_reason = (
            str(tracking.get("reason") or tracking.get("status") or "not ready")
            if not ready
            else f"stability_check_{ready_samples}/{stable_samples}"
        )
    except Exception as exc:
        last_reason = f"{type(exc).__name__}: {exc}"
    now = time.monotonic()
    if last_reason != last_reported_reason or now - last_report_at >= 5.0:
        print(f"Waiting for GB10: {last_reason}", flush=True)
        last_reported_reason = last_reason
        last_report_at = now
    time.sleep(0.25)
else:
    raise SystemExit(f"GB10 did not become ready within {timeout_s:.0f}s: {last_reason}")
PY

if ! kill -0 "${bridge_pid}" 2>/dev/null; then
  echo "Robot bridge exited before the GB10 became ready." >&2
  wait "${bridge_pid}"
  exit 1
fi

session_id="operator-bunny-test-$(date +%s)"
python3 - "${CLIENT_IP}" "${session_id}" <<'PY'
import json
import sys
import urllib.request

host, session_id = sys.argv[1:3]
with urllib.request.urlopen(f"http://{host}:8000/health", timeout=3) as response:
    health = json.load(response)
tracking = health.get("arm_tracking") or {}
calibration_id = str(tracking.get("calibration_id") or "")
if not calibration_id:
    raise SystemExit("GB10 did not report a calibration ID")
request = urllib.request.Request(
    f"http://{host}:8000/api/v1/arm/enable",
    data=json.dumps(
        {"session_id": session_id, "calibration_id": calibration_id}
    ).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=5) as response:
    report = json.load(response)
print(
    "LIVE TEST ENABLED: "
    f"state={report.get('state')} standing={report.get('standing')} "
    f"session={session_id}",
    flush=True,
)
PY
session_enabled=1

wait "${bridge_pid}"
