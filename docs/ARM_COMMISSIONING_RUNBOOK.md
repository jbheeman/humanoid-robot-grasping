# G1 right-arm commissioning runbook

Commissioning validates robot-local movement before camera tracking. The
robot ROS node owns all 14 arm command slots, holds the measured left arm, and
permits only guarded one-joint right-arm jogs. The first session is read-only.

The browser Stop control is a bounded software release, not a physical e-stop.

## 1. Read-only preflight

Start the robot commissioning node:

```bash
CLIENT_IP=<GB10_IP> \
G1_ROBOT_ID=g1-lab-01 \
ALLOW_MOVEMENT=0 \
uv run g1 arm commissioning
```

Start GB10 separately in dry-run mode:

```bash
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

Open `http://<GB10_IP>:8000/commissioning/`. Confirm:

- control mode is `commissioning` and movement is unavailable;
- the physical robot is the intended 29-DOF G1;
- right-arm joints occupy slots 22–28 and values are finite;
- `/lowstate` and `/g1/arm/state` are fresh;
- the actual motion-mode name is visible;
- motor temperature/lost-state fields are healthy;
- no unexpected `/arm_sdk` publisher exists.

Use `g1 inspect ros` on each host if any state is missing. Do not proceed by
disabling a gate.

## 2. Explicit movement launch

Stop the read-only robot node. Clear the exclusion zone, support the robot,
assign a spotter, and place the physical e-stop in hand. Restart with the exact
mode observed during preflight:

```bash
CLIENT_IP=<GB10_IP> \
G1_ROBOT_ID=g1-lab-01 \
EXPECTED_MOTION_MODE=<EXACT_MODE_FROM_PREFLIGHT> \
ALLOW_MOVEMENT=1 \
uv run g1 arm commissioning
```

This does not move or enable the arm. The node refuses movement if mode,
standing, state freshness, motor health, or exclusive ownership cannot be
verified. It never releases the robot motion mode automatically.

## 3. Create and enable a session

In the GB10 commissioning page:

1. Enter the operator name and type the displayed area-clear acknowledgement.
2. Create a short-lived in-memory session.
3. Enable at the measured pose.
4. Verify the weight ramp causes no visible position jump.
5. Test browser close or heartbeat loss and confirm the robot returns to
   `DISARMED` through the bounded release.

No bearer token, robot HTTP URL, or SSH tunnel is used. Off-LAN browsers may
tunnel GB10 port `8000` only.

## 4. Verify every joint

For each canonical right-arm joint:

1. Enable “record as sign check.”
2. Jog `+0.01 rad`; wait for `AWAITING_CONFIRM`.
3. Confirm only the intended joint moved and the encoder delta is positive,
   between `0.005` and `0.015 rad`.
4. Jog `-0.01 rad` to baseline and confirm.
5. Repeat in the negative direction and return.

The node rejects overlapping jogs and faults on nonselected/left-arm drift,
following error, stale state, stance loss, mode change, publisher conflict, or
motor-state failure.

## 5. Teach, replay, and promote

After all sign checks pass, teach one `0.01 rad` jog at a time. Keep each stage
within `0.05 rad` of its approved baseline and the session within `0.30 rad`
per joint of the measured start. Capture a candidate only when settled and
physically verified.

Replay the candidate twice from at least `0.02 rad` away. Then press Stop,
wait for `DISARMED` and zero weight, and promote. Promotion writes an atomic,
mode-0600, robot-bound profile at
`~/.config/g1-grasping/right-arm-home.json`; a hash mismatch fails closed.

Copy the profile and validated camera calibration to GB10 outside Git, then
run dry-run tracking:

```bash
ARM_HOME=/secure/g1-right-arm-home.json \
G1_ROBOT_ID=g1-lab-01 \
CALIBRATION=/secure/g1-camera.yaml \
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

The profile is an IK seed; it is not commanded automatically.

Session evidence remains under
`runs/research/arm_commissioning/<session-id>/` as manifest, event, telemetry,
and summary JSON. Never edit a promoted profile manually.
