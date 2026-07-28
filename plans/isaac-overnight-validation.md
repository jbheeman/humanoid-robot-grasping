# G1 interception: simulator-first overnight plan

## Outcome

By morning, first produce a reproducible pass/fail report for the exact physical
command replay. Then add synthetic left-to-right bunny lanes through the actual
localization/prediction/IK path. Do not connect to or command the physical G1
from the simulator. A joint replay is explicitly not an end-to-end interception
validation.

## Confirmed failure

- The touch run published targets at about 8 Hz in sampled telemetry and reached
  the bunny.
- Contact/hand occlusion caused eight consecutive `target_lost` samples over
  1.64 seconds.
- The bridge received no new command for 750 ms, entered `target_deadman`, then
  released.
- Command publication itself is cheap (sub-millisecond). Physical-run pipeline
  age is approximately 193 ms p50, 259 ms p95, and 288 ms p99.
- The calibrated localization contract is camera optical -> G1 torso
  (`x` forward, `y` left, `z` up). Camera-right motion therefore maps primarily
  to torso `-y`, the G1 right side.

## Sprint 1 — Host and replay parity

1. Use the replacement Isaac host at `theaa@10.0.0.65` over SSH port `6023`.
   The endpoint currently reports `Host is down`, so local implementation and
   tests continue until it is reachable.
2. Run `scripts/sim/probe-isaac-host.sh`.
3. Work under `~/Documents/coding/isaacsim/`. Locate or mount the external
   validation volume at `/data1/aarav`; do not silently redirect large assets
   or result sets when that mount is absent.
4. Select the installed native Windows Isaac Sim 6.0.1 / Isaac Lab v3.0.0-beta
   environment and pin Unitree's `unitree_sim_isaaclab` commit.
5. Export the touch run with `scripts/sim/export-g1-replay.py`.
6. Replay the complete 29-DOF measured initial state and right-arm targets at a 250 Hz
   physics/control step using the production Ruckig bridge.

Acceptance:

- Exact joint-name mapping is reported before playback.
- Table coordinates and object coordinates use the saved torso-frame
  calibration, not hand-tuned world coordinates.
- Replay never initializes DDS or ROS and cannot reach the physical robot.
- The calibrated frame resolves exactly to `torso_link`; pelvis and waist
  fallbacks are forbidden.
- Recorded pipeline ages are preserved and every rejected target code is
  reported.
- Simulated command position differs from the exported production command by
  at most 0.02 rad p95 before physics tracking error is applied.

## Sprint 2 — Physics/contact parity

1. Use Unitree's official fixed-base G1-29DOF-Dex1 USD and actuator parameters.
2. Build the 0.34 m x 0.68 m tabletop from the calibrated support region.
3. Use a soft-body proxy if available; otherwise use a mass/friction-matched
   capsule and document the approximation.
4. Record joint position/velocity, applied effort, hand-object distance,
   contact duration, object displacement, and controller state every step.

Acceptance:

- Gravity is enabled.
- The elbow does not sag more than 0.03 rad during a two-second static reach.
- Contact is detected before the physical replay loses vision.
- The baseline replay reproduces the deadman release after the recorded
  occlusion rather than hiding it.

## Sprint 3 — Candidate behavior matrix

This sprint requires a synthetic perception source that feeds the production
localization/prediction/IK path. Repositioning the bunny while replaying old
joint targets does not count.

Run deterministic seeds across:

- bunny lanes at torso `y = +0.20, +0.10, 0.00, -0.10 m`;
- left-to-right image motion, represented as torso `-y`, at 0.15, 0.30, and
  0.50 m/s;
- table near edge at torso `x = 0.24, 0.28, 0.32 m`;
- joint limits `(v, a, j)` of `(1,4,30)`, `(1.5,6,40)`, `(2,8,50)`;
- contact occlusion of 0.0, 0.3, 0.6, 1.0, and 1.5 seconds.

The production candidate must approach from the bunny's camera-right side:
target `y <= object y`, with a downstream `-y` offset selected using reachable
Ruckig arrival time. Initial table escape may still use the only certified near
edge; collision validation remains mandatory.

Acceptance:

- At least 95% contact rate over the matrix.
- No table/self collision and no joint-limit violation.
- Contact time p95 <= 1.5 seconds from the first stable observation.
- End-effector speed p95 improves over baseline without exceeding the chosen
  joint velocity/acceleration/jerk limits.
- A 1.0 second contact occlusion holds the already-reached blocking pose without
  retreating or chasing an unobserved bunny.
- PhysX contact force, not a link-origin distance threshold, establishes
  contact. Table and self-collision contacts are reported separately.

## Sprint 4 — Production patch gate

Only port a candidate back into runtime after it passes Sprint 3. The likely
patch is a bounded contact/occlusion latch plus explicit torso `-y` approach
bias; increasing Ruckig limits alone is not sufficient.

Acceptance:

- Existing unit tests pass.
- New tests cover transform direction, right-side bias, occlusion latch expiry,
  and unchanged collision rejection.
- One dry-run replay matches simulator-selected behavior before any operator
  chooses to run another physical test.

## Commands

```bash
# On the GB10: export the first episode in the current research session.
uv run python scripts/sim/export-g1-replay.py \
  runs/research/arm_tracking/20260727_222248_2549418/telemetry.jsonl \
  runs/sim/touch-run.json \
  --episode 0

# On the replacement Isaac workstation, after SSH is reachable.
ssh -p 6023 theaa@10.0.0.65
cd ~/Documents/coding/isaacsim/humanoid-robot-grasping
bash scripts/sim/probe-isaac-host.sh
bash scripts/sim/bootstrap-isaac-host.sh
```
