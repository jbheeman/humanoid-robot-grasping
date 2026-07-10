# RealSense-Guided G1 Arm Tracking Runbook

This workflow uses the robot RealSense Z16 stream as the only depth source. It does not use monocular or learned depth. The normal mode is dry-run; starting the services does not authorize movement.

## Architecture and ports

| Host | Port | Service |
| --- | ---: | --- |
| GB10 | 8000 | RGB inference, tracking status, MJPEG, and REST |
| GB10 | 8080 | Static viewer |
| Robot | 8765 | Existing locomotion bridge |
| Robot | 8766 | Authenticated persistent arm bridge |
| Robot | 8767 | RealSense depth health, calibration, and WebSocket |

The default robot address is `192.168.0.213`. Override it with `ROBOT_HOST` or the relevant CLI option.

## Install

On GB10:

```bash
uv run g1 setup gb10
```

This installs `vision`, `train`, and `arm` groups and fetches the Unitree sources pinned in `docs/PLAN.md`. The YOLO checkpoint is intentionally ignored by Git and must exist at:

```text
models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.pt
```

On the robot, install a system RealSense backend first (`realsense-ros`/`rclpy`, or `pyrealsense2`), then:

```bash
uv run g1 setup robot
```

The robot environment is isolated in `robot/` and rejects Torch or Ultralytics. It contains no training or CUDA stack.

## Calibration

The calibration CLI is replayable without hardware except for `probe`:

```bash
uv run --group arm python -m object_tracking.arm_tracking.calibration_cli probe \
  --output runs/calibration/camera-template.yaml

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli collect \
  runs/calibration/correspondences.yaml --subset solve --name solve-01 \
  --optical X Y Z --torso X Y Z --source apriltag

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli collect \
  runs/calibration/correspondences.yaml --subset validation --name validation-01 \
  --optical X Y Z --torso X Y Z --source apriltag

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli solve \
  runs/calibration/template.yaml runs/calibration/correspondences.yaml \
  runs/calibration/g1-camera.yaml

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli validate \
  runs/calibration/g1-camera.yaml --camera-serial SERIAL --waist-deg 0 0 0
```

Collect at least eight spatially distributed solve poses and four held-out validation poses. Attach an AprilTag rigidly to the right wrist. Manual points enter the same schema but do not bypass validation. Execution requires held-out RMSE at most `25 mm`, maximum error at most `50 mm`, orientation error at most `3°`, and waist deviation at most `3°`.

Bounded tuning is available, but creates a new timestamp/hash and re-runs residual validation:

```bash
uv run --group arm python -m object_tracking.arm_tracking.calibration_cli tune \
  runs/calibration/g1-camera.yaml runs/calibration/g1-camera-tuned.yaml \
  --xyz 0.001 0 0 --rpy-deg 0 0.1 0
```

## Start disarmed and run dry-run tracking

Create a bearer token on the robot without putting it in shell history:

```bash
umask 177
openssl rand -hex 32 > ~/.config/g1-arm-token
```

Start all robot services. The arm bridge is disarmed and movement-disabled:

```bash
CLIENT_IP=GB10_IP \
ARM_TOKEN_FILE="$HOME/.config/g1-arm-token" \
CALIBRATION=/path/to/g1-camera.yaml \
uv run g1 robot start
```

The depth service first tries `/camera/camera/aligned_depth_to_color/image_raw`, then direct depth-only `librealsense`. Direct depth is registered to RGB on GB10 with the stored intrinsics/extrinsics. If neither source opens—such as when `videohub_pc4` blocks depth—the service fails closed.

Copy the calibration and token to GB10 over a protected channel; keep both outside Git. Start the GB10 service in dry-run:

```bash
ROBOT_HOST=192.168.0.213 \
CALIBRATION=/secure/runtime/g1-camera.yaml \
uv run g1 gb10 start
```

When the GB10 server starts, it prints a copy-paste LAN URL and SSH tunnel command for the MacBook. Open `http://GB10:8080/unitree_dual_viewer.html?single=1`. The research console shows annotated RGB, hardware-depth colormap, camera/YOLO/encode rates, depth age and RGB/depth skew, detections and tracks, measured/predicted/target XYZ, IK errors, arm state and loop timing, rejection reasons, calibration identity, and research-session statistics.

Structured telemetry recording is enabled by default at 5 Hz. Each server start creates an isolated directory:

```text
runs/research/arm_tracking/YYYYMMDD_HHMMSS_PID/
  manifest.json
  telemetry.jsonl
  summary.json
```

This is separate from YOLO/VLM training outputs. It records structured observations and decisions, not raw images. Override it with `RESEARCH_ROOT`, change sampling with `RESEARCH_HZ`, or disable it with `RESEARCH_RECORD=0`.

Label controlled experiments with `RESEARCH_LABEL` and `RESEARCH_NOTES`, then analyze the newest session with `uv run g1-tune analyze runs/research/arm_tracking`. See [TUNING_GUIDE.md](TUNING_GUIDE.md) for the joint audit, repeatable scene protocol, safe parameter sweeps, and run comparison.

MacBook tunnel:

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -L 8000:127.0.0.1:8000 \
  -L 8766:192.168.0.213:8766 \
  -L 8767:192.168.0.213:8767 \
  USER@GB10_HOST
```

## REST and wire contracts

Depth:

- `GET :8767/health`
- `GET :8767/depth/calibration`
- `WS :8767/depth/stream`

GB10 research:

- `GET :8000/research/session`
- `GET :8000/research/summary`
- `GET :8000/research/telemetry?limit=100`
- `GET :8000/research/export.jsonl`

Each binary WebSocket message is a network-order 4-byte JSON-header length, a versioned JSON header, then zstd-compressed little-endian Z16. The decoder bounds header, compressed payload, dimensions, and decompressed size and verifies SHA-256.

Arm:

- `GET :8766/health`
- `GET :8766/state`
- Authenticated `POST :8766/arm/enable`
- Authenticated `POST :8766/arm/target`
- Authenticated idempotent `POST :8766/arm/stop`

Targets contain `session_id`, strictly increasing `sequence`, `calibration_id`, seven `right_arm_q` radians, and a Unix `source_timestamp`. Targets older than `250 ms`, discontinuous commands, invalid states, or mismatched sessions/calibrations are rejected.

## Physical movement is a separate operator gate

Code delivery does not claim physical movement acceptance. Before any movement, place the robot in a cleared exclusion zone with a spotter and physical e-stop, validate the depth/RGB registration board across center and corners, and confirm depth/arm health.

Restart only the arm bridge with explicit movement authorization and the validated calibration ID:

```bash
python scripts/robot/arm_bridge.py --allow-movement \
  --calibration /path/to/g1-camera.yaml --token-file "$HOME/.config/g1-arm-token"
```

Explicitly enable a short-lived session:

```bash
TOKEN="$(cat "$HOME/.config/g1-arm-token")"
curl -X POST http://192.168.0.213:8766/arm/enable \
  -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' \
  -d '{"session_id":"operator-session-UUID","calibration_id":"CALIBRATION-UUID"}'
```

Then restart GB10 with `EXECUTE=1` and the token file. Stop immediately with:

```bash
curl -X POST http://192.168.0.213:8766/arm/stop \
  -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' -d '{}'
```

Proceed in separate stop/go stages: depth-only validation, calibration validation, deadman test, at most `0.05 rad` right-arm smoke delta, live dry-run pregrasp, and finally the 20 cm stand-off reach. Independently measure wrist error; the final target is at most `5 cm` error with no safety violation.
