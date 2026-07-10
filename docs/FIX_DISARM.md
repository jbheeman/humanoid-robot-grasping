# Fixing a Disarmed G1 Arm Bridge

Use this guide when the robot-local arm bridge starts, `/health` and `/state`
return JSON, but the bridge remains `DISARMED` and prints repeated
`[ClientStub] send request error` messages.

## What is healthy and safe

The following state is a successful **read-only** bridge connection, not a
failure:

```json
{
  "ok": true,
  "state": "DISARMED",
  "allow_movement": false,
  "weight": 0.0,
  "robot_state_fresh": true,
  "controller_available": true
}
```

If `/state` includes fresh `measured_arm_q` values, the bridge is receiving
the robot's LowState DDS data. It is safe to use this state for camera/pose
visualization and GB10 dry-run tracking.

`DISARMED`, `allow_movement: false`, and `weight: 0.0` are intentional safety
gates. Do not bypass them to test video or detection.

## Check the local bridge first

Run these commands **on the robot** while the bridge process is running:

```bash
curl -s http://127.0.0.1:8766/health
curl -s http://127.0.0.1:8766/state
ip -br addr
ip route
```

For the current lab layout, the Unitree network interface is `wlan0` and the
robot address is `192.168.0.213`. Start the bridge explicitly on that
interface:

```bash
export CYCLONEDDS_HOME="$HOME/cyclonedds_ws/install/cyclonedds"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME"
export LD_LIBRARY_PATH="$HOME/humanoid-robot-grasping/robot/.venv/lib/python3.12/site-packages/unitree_sdk2py/utils/lib:$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

robot/.venv/bin/python scripts/robot/arm_bridge.py \
  --interface wlan0 \
  --domain-id 0 \
  --port 8766 \
  --token-file "$HOME/.config/g1-arm-token"
```

## About `ClientStub send request error`

The bridge uses a background Unitree `MotionSwitcherClient` request to check
the robot motion mode. A repeated request error means that this RPC check did
not receive a reply. It does **not** mean that the HTTP bridge, LowState DDS
subscription, or RGB relay has failed.

When `/state` still reports fresh arm positions and a motion mode name such as
`ai`, the read-only bridge is usable. The error becomes a movement blocker only
when commissioning or execution requires verified motion mode and motor state.

Common causes to investigate before movement:

- wrong DDS interface or domain;
- Unitree motion-switcher service not running or not reachable;
- another controller owns the arm service;
- the robot is not in a compatible standing/motion mode;
- unavailable motor status fields.

## Before authorizing any movement

Do this only with a cleared exclusion zone, spotter, and physical e-stop:

1. Run the separate commissioning workflow in
   [ARM_COMMISSIONING_RUNBOOK.md](ARM_COMMISSIONING_RUNBOOK.md).
2. Start with the observed mode name:

   ```bash
   EXPECTED_MOTION_MODE=ai \
   ALLOW_MOVEMENT=1 \
   COMMISSIONING_ACK="I HAVE A SPOTTER AND PHYSICAL E-STOP" \
   uv run g1 arm commissioning
   ```

3. Confirm fresh LowState, verified controller ownership, healthy motor state,
   and the exact motion mode before enabling a session.
4. Use the 0.01 rad supervised sign checks before any tracking command.

GB10 `--dry-run` tracking and the browser visualization do not need these
movement gates. They should be the first stage of every camera test.
