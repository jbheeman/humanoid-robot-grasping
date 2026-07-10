# G1 Right-Arm Commissioning Runbook

Commissioning validates robot-local joint movement before camera calibration, IK, or object tracking. The 250 Hz `rt/arm_sdk` bridge remains on the robot, commands all 14 arm slots, holds the measured left-arm pose, and permits only one-joint right-arm jogs.

The first session is read-only. Physical movement requires a stable standing robot, cleared exclusion zone, spotter, and physical e-stop. The webpage Stop control is only a bounded software release.

## Topology

Initial commissioning:

```text
G1 robot: LowState + 250 Hz arm bridge + tiny event log
       │ SSH tunnel
       ▼
MacBook: browser wizard or g1-arm CLI
```

Later tracking:

```text
G1 robot: RGB/depth relay + 250 Hz arm bridge
       │
       ▼
GB10: YOLO, depth fusion, IK, telemetry, website
       │
       ▼
MacBook: browser only
```

GB10 never publishes DDS directly. It sends high-level targets to the robot-local bridge. To route the commissioning page through GB10, bind the robot bridge to its trusted robot-network address and tunnel port 8766 through GB10; never expose it to an untrusted network.

## 1. Read-only preflight

On the robot, provision the mode-0600 bearer token and start commissioning without movement:

```bash
ARM_TOKEN_FILE="$HOME/.config/g1-arm-token" \
G1_ROBOT_ID="g1-lab-01" \
ALLOW_MOVEMENT=0 \
uv run g1 arm commissioning
```

On the MacBook:

```bash
ssh -N -L 8766:127.0.0.1:8766 USER@ROBOT_IP
```

Open `http://127.0.0.1:8766/commissioning/`, enter the bearer token, and press Connect. Confirm:

- control mode is `commissioning`;
- the physical robot is the 29-DOF G1 variant;
- seven right-arm joints appear in SDK slots 22–28;
- measured positions and velocities are finite;
- LowState is fresh;
- the actual motion-mode name is displayed;
- motor temperature/lost-status fields are available;
- the bridge reports no existing `rt/arm_sdk` publisher conflict.

Read-only preflight cannot enable because `--allow-movement` is absent.

## 2. Explicit movement startup

Stop the read-only bridge. Restart using the exact motion-mode name observed during preflight:

```bash
ARM_TOKEN_FILE="$HOME/.config/g1-arm-token" \
G1_ROBOT_ID="g1-lab-01" \
EXPECTED_MOTION_MODE="EXACT_MODE_FROM_PREFLIGHT" \
ALLOW_MOVEMENT=1 \
COMMISSIONING_ACK="I HAVE A SPOTTER AND PHYSICAL E-STOP" \
uv run g1 arm commissioning
```

The process refuses movement without the expected mode and startup acknowledgment. It never calls `MotionSwitcher.ReleaseMode()`.

## 3. Create and enable one session

In the wizard:

1. Enter the token and operator name.
2. Type `I HAVE CLEARED THE ROBOT AREA` exactly.
3. Create the session.
4. Enable at the measured pose.
5. Verify the weight ramp causes no visible position jump.

The browser sends a 10 Hz heartbeat after enable. Closing the page, losing the network, or stopping the heartbeat triggers the 500 ms deadman and 250 ms weight release. The first real test should also run the CLI watcher in a terminal so Ctrl-C sends Stop:

```bash
uv run g1-arm --url http://127.0.0.1:8766 \
  --token-file /secure/local/g1-arm-token watch SESSION_ID
```

## 4. Verify every joint

Enable “Record jog as sign check.” For each canonical right-arm joint:

1. Jog `+0.01 rad` and wait for `AWAITING_CONFIRM`.
2. Confirm that only the intended joint moved; encoder delta must be positive and between 0.005 and 0.015 rad.
3. Jog `-0.01 rad` back to baseline and confirm.
4. Repeat in the negative direction and return.

The server rejects another jog while moving or waiting for confirmation. It faults on nonselected-joint drift above 0.01 rad, left-arm drift above 0.01 rad, following error above 0.05 rad, stale LowState, stance loss, motion-mode change, controller conflict, or motor-state failure.

## 5. Teach a candidate pose

After all fourteen sign checks pass, disable sign-check mode and teach with one `0.01 rad` jog at a time. The server enforces:

- at most 0.05 rad from the current approved stage baseline;
- explicit “Approve stage” before rebasing the stage;
- at most 0.30 rad per joint from the session’s measured starting pose;
- 0.10 rad/s velocity and 0.50 rad/s² acceleration;
- five-second motion timeout;
- velocity below 0.02 rad/s for 0.5 seconds before confirmation.

Capture `safe_chest` only when the arm is settled and the operator has verified physical clearance. V1 stops at the 0.30 rad envelope even if this does not produce a full chest pose.

## 6. Replay and promote

To validate the candidate:

1. Jog at least 0.02 rad away from it.
2. Use “Replay one 0.01 rad step” until measured values return to the candidate.
3. Validate the replay.
4. Repeat the departure and replay a second time.
5. Press Software Stop and wait for `DISARMED` with zero weight.
6. Promote the profile.

Promotion writes an atomic, mode-0600 robot-specific profile to `$HOME/.config/g1-grasping/right-arm-home.json`. It stores measured positions, joint order/contract, robot identity, sign-check evidence, replay count, and a content hash. It never changes tracking automatically.

Copy the promoted profile to GB10 over a protected channel and opt into it during dry-run:

```bash
ARM_HOME=/secure/runtime/g1-right-arm-home.json \
G1_ROBOT_ID="g1-lab-01" \
ROBOT_HOST=ROBOT_IP \
CALIBRATION=/secure/runtime/g1-camera.yaml \
uv run g1 gb10 start
```

Tracking uses the profile only as an IK seed. It does not command the pose or bypass camera calibration.

## Research output

Each session writes separately from training and tracking runs:

```text
runs/research/arm_commissioning/<session-id>/
  manifest.json
  events.jsonl
  telemetry.jsonl
  summary.json
```

Events include the measured start/end, target, selected joint, motion type, encoder confirmation, peak error, checkpoints, stops, and faults. Never edit a promoted profile manually; a hash mismatch fails closed.
