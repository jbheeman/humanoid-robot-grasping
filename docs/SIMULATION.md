# Suspended G1 arm simulation

This branch tunes the seven right-arm joints against the official Unitree
`g1_29dof.xml` in MuJoCo 3.3.6. The pelvis is welded to emulate the lab hanger.
The model is composed from many meshes and bodies; it is not one merged STL.
Simulation reads no robot network state and can never publish a Unitree command.

## Setup and smoke test

```bash
scripts/sim/setup.sh
scripts/sim/run.sh doctor
scripts/sim/run.sh smoke
```

The official model is sparse-checked out at commit
`ae6a8403e272733e9996ef59990880330496177f` into ignored `.deps/`. The
isolated Python environment lives in ignored `.sim-venv/`.

## Long tuning run

```bash
nohup scripts/sim/run.sh sweep --workers 14 --min-hours 48 --max-hours 72 \
  --plateau-hours 12 >runs/simulation-runner.log 2>&1 &
scripts/sim/run.sh status
```

The sweep tests every right-arm joint in both directions at 0.01, 0.03, and
0.05 rad with randomized damping and gravity. It searches bridge-like
`kp`, `kd`, velocity, and acceleration limits. A candidate is penalized
for p95 following error, overshoot, failure to settle, excess non-selected
joint drift, and torque saturation. The sweep stops no earlier than 48 hours,
no later than 72 hours, and after 48 hours may stop when no candidate improves
the robust score by at least 1% for 12 hours.

Raw checkpoints stay under ignored `runs/simulation/`. Only compact best
candidate and validation summaries are promoted under
`simulation/baselines/g1_right_arm/`.

## Safety and real tuning

Encoder positions and velocities are already the measured feedback used by the
real arm bridge. Simulation gains are advisory and are never applied
automatically. Before real use, replay a guarded 0.01 rad commissioning jog for
each joint, capture encoder response, compare p95 error/overshoot/settling and
torque, then promote one change at a time. Keep the existing arm authorization,
limit, slew, following-error, drift, and e-stop gates enabled.

## Encoder calibration and guarded validation

Simulation does not replace measured encoder validation. The existing guarded
commissioning recorder writes `telemetry.jsonl` at 20 Hz with 14-arm bridge
vectors; the calibration command automatically selects the seven right-arm
slots. After a guarded session, run:

```bash
scripts/sim/run.sh calibrate +  runs/research/arm_commissioning/SESSION_ID/telemetry.jsonl
```

Review the generated RMSE and p95 errors before changing any physical gains.
Use 0.01 rad guarded jogs first, one joint at a time, with the existing
authorization, following-error, drift, and emergency-stop gates enabled.
The simulator deliberately prioritizes shoulders and elbow because they are
the current high-error links; wrists remain in the evaluation but have lower
ranking weight.
