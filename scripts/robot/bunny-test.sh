#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENT_IP="${CLIENT_IP:-192.168.0.66}"
CALIBRATION="${CALIBRATION:-${ROOT_DIR}/runs/localization/g1-tabletop-calibration.json}"
EXPECTED_MOTION_MODE="${EXPECTED_MOTION_MODE:-ai}"
MAX_TILT_DEG="${MAX_TILT_DEG:-8}"
MAX_WAIST_DEVIATION_DEG="${MAX_WAIST_DEVIATION_DEG:-12}"
ARM_WEIGHT_RAMP_S="${ARM_WEIGHT_RAMP_S:-0.75}"
ARM_MAX_VELOCITY_RAD_S="${ARM_MAX_VELOCITY_RAD_S:-1.00}"
ARM_MAX_ACCELERATION_RAD_S2="${ARM_MAX_ACCELERATION_RAD_S2:-4.00}"
ARM_MAX_JERK_RAD_S3="${ARM_MAX_JERK_RAD_S3:-30.0}"
ARM_MAX_FOLLOWING_ERROR_RAD="${ARM_MAX_FOLLOWING_ERROR_RAD:-0.35}"
FOLLOW_PROFILE="${FOLLOW_PROFILE:-balanced}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-90}"
READY_STABLE_SAMPLES="${READY_STABLE_SAMPLES:-4}"
# RGB uses the root-owned 960x540@60 GStreamer service. Librealsense opens only
# the depth interface so NVENC and ROS depth serialization cannot block the
# same capture loop.
DEPTH_CAPTURE_FPS="${DEPTH_CAPTURE_FPS:-60}"
DEPTH_PUBLISH_FPS="${DEPTH_PUBLISH_FPS:-20}"
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
    --follow-profile)
      FOLLOW_PROFILE="${2:?--follow-profile requires balanced or aggressive}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: uv run g1 robot bunny-test [--client-ip 192.168.0.66]

Starts the foreground robot bridge in DISARMED state, requires an interactive
operator confirmation, then verifies fresh RGB/depth, bunny detection, table
geometry, and safe IK before enabling one guarded tracking session. Ctrl-C
retraces the accepted arm path to the measured startup pose before release.
EOF
      exit 0
      ;;
    *)
      echo "Unknown bunny-test option: $1" >&2
      exit 2
      ;;
  esac
done

case "${FOLLOW_PROFILE}" in
  balanced)
    ARM_MAX_VELOCITY_RAD_S="1.00"
    ARM_MAX_ACCELERATION_RAD_S2="4.00"
    ARM_MAX_JERK_RAD_S3="30.0"
    ARM_MAX_FOLLOWING_ERROR_RAD="0.35"
    ;;
  aggressive)
    ARM_MAX_VELOCITY_RAD_S="1.25"
    ARM_MAX_ACCELERATION_RAD_S2="5.00"
    ARM_MAX_JERK_RAD_S3="40.0"
    ARM_MAX_FOLLOWING_ERROR_RAD="0.40"
    ;;
  *)
    echo "FOLLOW_PROFILE must be balanced or aggressive." >&2
    exit 2
    ;;
esac

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
NATIVE_ARM_SOCKET="${NATIVE_ARM_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/g1-native-arm-${UID}.sock}"
return_remote_arm() {
  if [[ "${session_enabled}" == "1" ]]; then
    python3 - "${CLIENT_IP}" "${NATIVE_ARM_SOCKET}" <<'PY' || true
import json
import socket
import sys
import time
import urllib.request

host, socket_path = sys.argv[1:3]


def native_call(operation, payload):
    envelope = {
        "request_id": f"bunny-cleanup-{time.monotonic_ns()}",
        "operation": operation,
        "payload": payload,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        client.connect(socket_path)
        client.sendall(
            (json.dumps(envelope, separators=(",", ":")) + "\n").encode()
        )
        response = b""
        while not response.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
    decoded = json.loads(response)
    if not decoded.get("ok"):
        raise RuntimeError(decoded.get("message") or "native arm request failed")
    return decoded.get("report") or {}


reason = "operator_bunny_test_interrupted"
try:
    request = urllib.request.Request(
        f"http://{host}:8000/api/v1/arm/return-to-neutral",
        data=json.dumps({"reason": reason}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        report = json.load(response)
except Exception as exc:
    print(f"GB10 return request failed ({exc}); using robot-local bridge.", flush=True)
    try:
        report = native_call("return", {"reason": reason})
    except Exception as local_exc:
        print(f"Controlled return unavailable: {local_exc}", flush=True)
        try:
            native_call("stop", {"reason": "return_request_failed"})
        except Exception:
            pass
        raise SystemExit(1)

print(
    f"Returning arm to measured startup pose: state={report.get('state')}",
    flush=True,
)
deadline = time.monotonic() + 12.5
while time.monotonic() < deadline:
    try:
        report = native_call("state", {})
    except Exception:
        time.sleep(0.1)
        continue
    state = report.get("state")
    if state == "DISARMED":
        print("Arm returned to startup pose and released.", flush=True)
        raise SystemExit(0)
    if state == "FAULT":
        print(
            f"Return faulted: {report.get('fault_reason')}; forcing bounded release.",
            flush=True,
        )
        try:
            native_call("stop", {"reason": "return_fault"})
        except Exception:
            pass
        raise SystemExit(1)
    time.sleep(0.1)

print("Controlled return timed out; forcing bounded release.", flush=True)
try:
    native_call("stop", {"reason": "return_cleanup_timeout"})
except Exception:
    pass
raise SystemExit(1)
PY
  fi
}
cleanup() {
  if [[ "${cleaning_up}" == "1" ]]; then
    return
  fi
  cleaning_up=1
  return_remote_arm
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
echo "  following error: ${ARM_MAX_FOLLOWING_ERROR_RAD} rad"
echo "  follow profile: ${FOLLOW_PROFILE}"
echo "  weight ramp: ${ARM_WEIGHT_RAMP_S} s"
echo "  depth rate:  ${DEPTH_CAPTURE_FPS} Hz"
echo "  depth publish: ${DEPTH_PUBLISH_FPS} Hz"
echo "  ROS domain:  ${PROJECT_ROS_DOMAIN_ID}"
echo "Keep the physical E-stop in hand. Ctrl-C returns to startup pose, then releases."

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
ARM_MAX_FOLLOWING_ERROR_RAD="${ARM_MAX_FOLLOWING_ERROR_RAD}" \
DEPTH_CAPTURE_FPS="${DEPTH_CAPTURE_FPS}" \
DEPTH_PUBLISH_FPS="${DEPTH_PUBLISH_FPS}" \
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

if [[ ! -t 0 ]]; then
  echo "Refusing to enable: bunny-test requires an interactive terminal." >&2
  echo "Run the command manually and press Enter at the operator gate." >&2
  exit 1
fi
echo
echo "ARM IS DISARMED. Confirm the area is clear and keep the E-stop in hand."
read -r -p "Press Enter to begin GB10 verification and guarded arm enable... " _
echo "Operator confirmed. Beginning guarded readiness verification."

python3 - "${CLIENT_IP}" "${READY_TIMEOUT_S}" "${READY_STABLE_SAMPLES}" "${FOLLOW_PROFILE}" <<'PY'
import json
import sys
import time
import urllib.request

host = sys.argv[1]
timeout_s = float(sys.argv[2])
stable_samples = int(sys.argv[3])
expected_follow_profile = sys.argv[4]
if stable_samples < 1:
    raise SystemExit("READY_STABLE_SAMPLES must be positive")
url = f"http://{host}:8000/health"
deadline = time.monotonic() + timeout_s
last_reason = "GB10 has not responded"
last_reported_reason = None
last_report_at = 0.0
ready_samples = 0
profile_negotiated = False
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            health = json.load(response)
        tracking = health.get("arm_tracking") or {}
        with urllib.request.urlopen(
            f"http://{host}:8000/api/v1/arm/state", timeout=2
        ) as response:
            robot_arm = json.load(response)
        if not profile_negotiated:
            request = urllib.request.Request(
                f"http://{host}:8000/api/v1/arm/follow-profile",
                data=json.dumps(
                    {"follow_profile": expected_follow_profile}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                applied_profile = json.load(response)
            if applied_profile.get("follow_profile") != expected_follow_profile:
                raise RuntimeError("GB10 did not apply the robot-selected follow profile")
            print(
                "GB10 follow profile selected by robot: "
                f"{expected_follow_profile}",
                flush=True,
            )
            profile_negotiated = True
            ready_samples = 0
            continue
        visualization = tracking.get("visualization") or {}
        support_status = visualization.get("support_plane_status") or {}
        support_source = str(support_status.get("source") or "")
        support_source_ready = (
            support_source.startswith("automatic_rgbd_")
            or support_source == "live_plane_calibrated_near_edge_dimension_prior"
        )
        command = tracking.get("predicted_bounded_arm_command_rad")
        structurally_ready = (
            tracking.get("mode") == "execute"
            and tracking.get("arm_state") == "DISARMED"
            and robot_arm.get("enable_ready") is True
            and tracking.get("ik_status") == "ok"
            and support_status.get("available") is True
            and support_source_ready
            and isinstance(command, list)
            and len(command) == 7
            and tracking.get("reason") == "arm_not_explicitly_enabled"
            and tracking.get("follow_profile") == expected_follow_profile
        )
        ready = (
            structurally_ready
            and float(tracking.get("depth_age_ms", 1e9)) <= 200.0
            and float(tracking.get("processing_latency_ms", 1e9)) <= 300.0
        )
        if ready:
            ready_samples += 1
        elif not structurally_ready and tracking.get("reason") not in {
            "depth_receive_timeout",
            "rgb_depth_pair_stale",
        }:
            # A missed asynchronous pair or one slow plane-refresh cycle does
            # not invalidate prior fresh IK samples. Genuine planning or robot
            # state failures still clear the readiness evidence.
            ready_samples = 0
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
        if not ready and robot_arm.get("enable_ready") is not True:
            blockers = robot_arm.get("enable_blockers") or ["unknown"]
            last_reason = "robot_enable_blocked:" + ",".join(
                str(item) for item in blockers
            )
        elif not ready:
            last_reason = str(tracking.get("reason") or tracking.get("status") or "not ready")
        else:
            last_reason = f"stability_check_{ready_samples}/{stable_samples}"
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
python3 - "${CLIENT_IP}" "${session_id}" "${FOLLOW_PROFILE}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host, session_id, follow_profile = sys.argv[1:4]
with urllib.request.urlopen(f"http://{host}:8000/health", timeout=3) as response:
    health = json.load(response)
tracking = health.get("arm_tracking") or {}
calibration_id = str(tracking.get("calibration_id") or "")
if not calibration_id:
    raise SystemExit("GB10 did not report a calibration ID")
request = urllib.request.Request(
    f"http://{host}:8000/api/v1/arm/enable",
    data=json.dumps(
        {
            "session_id": session_id,
            "calibration_id": calibration_id,
            "follow_profile": follow_profile,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        report = json.load(response)
except urllib.error.HTTPError as exc:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        detail = json.loads(raw).get("detail", raw)
    except json.JSONDecodeError:
        detail = raw
    raise SystemExit(f"GB10 arm enable failed (HTTP {exc.code}): {detail}") from exc
print(
    "LIVE TEST ENABLED: "
    f"state={report.get('state')} standing={report.get('standing')} "
    f"session={session_id}",
    flush=True,
)
PY
session_enabled=1

wait "${bridge_pid}"
