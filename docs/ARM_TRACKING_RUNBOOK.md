# RealSense-guided G1 arm tracking runbook

Tracking uses hardware depth, ROS 2 between robot and GB10, and dry-run by
default. Startup never authorizes movement.

## Install and calibrate

Run `bash scripts/robot/setup.sh` on Foxy and
`bash scripts/gb10/setup.sh` on Jazzy. The robot environment excludes FastAPI,
Torch, Ultralytics, and Unitree SDK2 Python.

The detector checkpoint is ignored by Git and must exist on GB10. Use the
calibration CLI to probe, collect solve/validation sets, solve, and validate:

```bash
uv run --group arm python -m object_tracking.arm_tracking.calibration_cli probe \
  --output runs/calibration/camera-template.yaml

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli collect \
  runs/calibration/correspondences.yaml --subset solve --name solve-01 \
  --optical X Y Z --torso X Y Z --source apriltag

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli solve \
  runs/calibration/template.yaml runs/calibration/correspondences.yaml \
  runs/calibration/g1-camera.yaml

uv run --group arm python -m object_tracking.arm_tracking.calibration_cli validate \
  runs/calibration/g1-camera.yaml --camera-serial SERIAL --waist-deg 0 0 0
```

Collect at least eight spatially distributed solve poses and four held-out
validation poses. Execution requires held-out RMSE at most `25 mm`, maximum
error at most `50 mm`, orientation error at most `3°`, and waist deviation at
most `3°`.

## Start read-only/dry-run

Robot:

```bash
CALIBRATION=/secure/g1-camera.yaml \
uv run g1 robot start
```

GB10:

```bash
CALIBRATION=/secure/g1-camera.yaml \
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

Open `http://<GB10_IP>:8000/`. The UI shows RGB, hardware depth,
detections/tracks, measured and predicted target position, IK, arm state,
timing, rejection reasons, calibration identity, and research status.

Depth is published on `/g1/depth` as bounded zstd-compressed Z16 with sensor
timestamp, dimensions, scale, calibration ID, registration flag, and checksum.
The subscriber keeps only the newest frame. RGB/depth pairing carries pipeline
age to the robot so the deadman does not rely on synchronized wall clocks.

Recording defaults to 5 Hz under
`runs/research/arm_tracking/<timestamp>/`. Set `RESEARCH_LABEL`,
`RESEARCH_NOTES`, `RESEARCH_HZ`, or `RESEARCH_RECORD=0` as needed.

## Health checks

Before movement, confirm:

```bash
uv run g1 inspect ros --role robot --peer <GB10_IP> --interface wlan0
uv run g1 inspect ros --role gb10 --peer <ROBOT_IP>
```

Then test stale depth, stale target, duplicate publisher, ROS link loss, robot
node exit, and GB10 shutdown. Every case must remain disarmed or perform the
bounded zero-weight release.

UI-only health/research endpoints remain on GB10 port `8000`; robot REST and
WebSocket contracts no longer exist. Do not use ports `8765–8767`, a token,
or a depth WebSocket URL.

## Physical movement gate

Complete [arm commissioning](ARM_COMMISSIONING_RUNBOOK.md) first. With a
cleared area, spotter, supported robot, physical e-stop, validated calibration,
and verified home profile, launch the robot with movement permission and the
observed mode:

```bash
CALIBRATION=/secure/g1-camera.yaml \
G1_ROBOT_ID=g1-lab-01 \
EXPECTED_MOTION_MODE=ai \
ALLOW_MOVEMENT=1 \
uv run g1 robot start --client-ip <GB10_IP>
```

Launch GB10 execution separately:

```bash
CALIBRATION=/secure/g1-camera.yaml \
ARM_HOME=/secure/g1-right-arm-home.json \
G1_ROBOT_ID=g1-lab-01 \
uv run g1 gb10 start --robot-host <ROBOT_IP> --execute
```

Enable through the GB10 UI. Proceed in stop/go stages: depth registration,
calibration validation, deadman test, at most `0.05 rad` smoke delta, live
dry-run pregrasp, then a 20 cm stand-off reach. Independently measure wrist
error; acceptance is at most `5 cm` with no safety violation.

Simulation may screen broad parameter sweeps, but it is not required for
normal commissioned arm movement and cannot replace these physical checks.
