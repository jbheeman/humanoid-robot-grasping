# Unitree G1 Movement Notes

These notes come from the copied SDK examples under `scripts/unitree_examples/` and the local wrappers in `scripts/`.

## High-Level Arm Gestures

`scripts/unitree_examples/g1/high_level/g1_arm_action_example.py` uses:

- `ChannelFactoryInitialize(0, <interface>)`
- `G1ArmActionClient`
- `action_map`
- `ExecuteAction(action_map.get("<name>"))`

Interesting action names already present in the SDK example:

- `release arm`
- `shake hand`
- `high five`
- `high wave`
- `face wave`
- `hands up`
- `right hand up`

For gestures like handshake or wave, prefer this high-level client first. It is closer to what Unitree ships and likely coordinates more hidden robot state than our custom low-level commands.

Local wrapper path:

```bash
python3 scripts/robot.py sdk-example g1_arm_action "shake hand"
python3 scripts/robot.py sdk-example g1_arm_action "high wave"
python3 scripts/robot.py sdk-example g1_arm_action "release arm"
```

## Low-Level Arm Calibration

The copied `g1_arm5_sdk_dds_example.py` and `g1_arm7_sdk_dds_example.py` do not publish arm commands to `rt/lowcmd`. They publish `LowCmd_` messages to:

```text
rt/arm_sdk
```

They also set motor command index `29` as an arm SDK enable weight:

```text
motor_cmd[29].q = 1  # enable arm_sdk
motor_cmd[29].q = 0  # disable arm_sdk
```

The examples use a 20 ms control loop, read current positions from `rt/lowstate`, set `tau = 0`, `dq = 0`, and write position targets with gains around:

```text
kp = 60.0
kd = 1.5
```

For our calibration branch, lower starting gains and smaller per-tick steps are safer for tuning. The `calibrate_arms` command uses feedback from `rt/lowstate`, publishes through `rt/arm_sdk`, and releases arm SDK control before returning.

## Joint Index Notes

The SDK examples identify these useful arm indices:

- Left shoulder pitch: `15`
- Left shoulder roll: `16`
- Left shoulder yaw: `17`
- Left elbow: `18`
- Left wrist roll: `19`
- Left wrist pitch: `20` on 29-DOF G1, invalid on 23-DOF G1
- Left wrist yaw: `21` on 29-DOF G1, invalid on 23-DOF G1
- Right shoulder pitch: `22`
- Right shoulder roll: `23`
- Right shoulder yaw: `24`
- Right elbow: `25`
- Right wrist roll: `26`
- Right wrist pitch: `27` on 29-DOF G1, invalid on 23-DOF G1
- Right wrist yaw: `28` on 29-DOF G1, invalid on 23-DOF G1
- Arm SDK enable weight: `29`

Finger control is not obvious from these copied examples. Treat finger joints as configuration-specific until we confirm the hand package and joint IDs on the actual robot. Use repeated `--joint-target ID:RAD` values only after confirming the ID exists in `rt/lowstate` and the robot is physically clear.

## Recommended Test Order

1. Start bridge read-only and verify health/probe.
2. Run `sdk-example g1_arm_action "release arm"`.
3. Run one high-level gesture such as `high wave` or `shake hand`.
4. On `calibration`, run `calibrate_arms` with a small `--arm-scale`, short hold, and no custom finger targets.
5. Only after low-state reports confirm available joint IDs, add explicit `--joint-target ID:RAD` overrides.

Avoid interactive keyboard control. Use bounded one-shot commands and stop between tests.
