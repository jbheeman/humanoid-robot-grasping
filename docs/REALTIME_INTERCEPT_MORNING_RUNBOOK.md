# Real-time bunny interception morning runbook

This runbook begins disarmed and dry-run. Do not skip directly to moving-bunny
execution. Keep a spotter and physical e-stop at the robot.

## 1. Pin the deployment

Run on the robot and GB10 from the project checkout:

```bash
git rev-parse HEAD
git status --short
```

Both hosts must report the same commit. Record hashes for:

```bash
sha256sum \
  /secure/g1-camera.yaml \
  /secure/g1-right-arm-home.json \
  /secure/g1-demo-intercept.yaml \
  models/plushie_detector/*/weights/best.engine
```

Copy `configs/intercept/demo-lane.example.yaml` to
`/secure/g1-demo-intercept.yaml`. Replace the calibration ID, measured lane
plane/bounds, measured ready-arm joints, physically checked palm-facing normal,
and conservative measured timing/motion values. Leave
`validated_for_execution: false`.

## 2. Start observations only

Robot:

```bash
uv run g1 robot start \
  --client-ip <GB10_IP> \
  --calibration /secure/g1-camera.yaml
```

GB10, alpha-beta baseline:

```bash
TRAJECTORY_MODEL= \
uv run g1 gb10 start \
  --robot-host <ROBOT_IP> \
  --calibration /secure/g1-camera.yaml \
  --arm-home /secure/g1-right-arm-home.json \
  --robot-id g1-lab-01 \
  --intercept-config /secure/g1-demo-intercept.yaml \
  --research-label intercept_passive_baseline \
  --dry-run
```

Open `http://<GB10_IP>:8000/` and inspect:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

No target may be published in dry-run.

After changing the lane setup or after a terminal `HOLD`/`EXPIRED` result,
explicitly reset acquisition:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/intercept/reset
```

Reset never enables the arm and never publishes a target.
In execute mode, a reset also releases the current arm session. Use the normal
arm-enable control again only after the rejection cause is understood.

## 3. Verify timing and current detections

Warm the detector, then perform at least ten passive lane crossings. Include
one empty-lane interval and one brief occlusion.

Required evidence:

- `rgb_frame_id` increases.
- Detector-frame age refers to the processed frame, not the latest raw frame.
- Retained tracks with missed detections are not arm observations.
- RGB/depth pair skew stays under 100 ms.
- Pipeline age stays below the robot's 250 ms target TTL.
- Intercept state progresses from `ACQUIRING` to `PREVIEW` and, near the
  crossing, `COMMITTED`.
- The committed Cartesian target remains fixed while fresh observations only
  reconfirm it.
- Target loss before commit produces no movement.
- Occlusion after commit expires no later than 250 ms after the last confirming
  observation.

Analyze the recorded run:

```bash
uv run g1 tune analyze runs/research/arm_tracking
```

Do not raise stale-pair or target-TTL limits to improve readiness.

## 4. Commission static motion

Complete `docs/ARM_COMMISSIONING_RUNBOOK.md`:

1. Confirm joint order/sign with one `0.01 rad` jog at a time.
2. Verify stop and bounded release.
3. Verify the robot-specific home profile.
4. Move to the proposed ready pose using the commissioned static path.
5. Physically confirm the palm faces the incoming lane.
6. Confirm measured right-arm joints are within
   `ready_pose_tolerance_rad`.
7. Preview the static intercept point and independently measure wrist error.
8. Verify workspace, tabletop clearance, next-edge collision result, following
   error, and no saturation.

Do not use the moving bunny for the first execution of the new path.

## 5. Measure empirical timing

From the ready pose, execute supervised small Cartesian steps comparable to the
expected intercept displacement. Record:

- target publication to robot acceptance;
- first measured motion;
- settle time;
- maximum following error;
- command delta, velocity, and acceleration limiting;
- distance travelled.

Replace the profile's planner speed, acceleration, compute delay, command
delay, settle time, and minimum deadline slack with conservative bounds from
these trials. Re-run dry-run. Only then set:

```yaml
validated_for_execution: true
```

## 6. Test loss behavior before contact

With movement permission but no bunny contact, inject:

1. target loss;
2. depth loss;
3. GB10/robot ROS interruption.

Each must stop or perform the existing bounded release. A stale observation
must never refresh a committed target.

## 7. Enable the staged live test

Robot:

```bash
uv run g1 robot start \
  --client-ip <GB10_IP> \
  --calibration /secure/g1-camera.yaml \
  --allow-movement \
  --expected-motion-mode ai
```

GB10:

```bash
TRAJECTORY_MODEL= \
uv run g1 gb10 start \
  --robot-host <ROBOT_IP> \
  --calibration /secure/g1-camera.yaml \
  --arm-home /secure/g1-right-arm-home.json \
  --robot-id g1-lab-01 \
  --intercept-config /secure/g1-demo-intercept.yaml \
  --research-label intercept_execute_005mps \
  --execute
```

Start with a measured 0.05 m/s bunny crossing. Run ten trials. Promote to the
next speed only after at least 9/10 clean blocks, with:

- zero prohibited contact;
- zero stale command acceptance;
- zero joint/velocity/acceleration saturation;
- positive conservative arrival slack for every command;
- no unexpected motion on rejected attempts;
- no post-crossing reversal.

Freeze at the highest proven speed.

## 8. Stop and preserve evidence

Stop through the API before shutting down processes:

```bash
curl -sS -X POST \
  -H 'content-type: application/json' \
  -d '{"reason":"operator_intercept_test_complete"}' \
  http://127.0.0.1:8000/api/v1/arm/stop
```

Preserve the run directory, exact configuration, commit, artifact hashes, and
synchronized trial video.

## Important exclusions

- Do not run `uv run g1 robot bunny-test` until static commissioning and all
  loss/deadman tests pass.
- Do not use `scripts/robot/direct_plushie_track.py`.
- Do not enable a GRU/VLA during baseline tuning.
- Do not change wrist orientation in the real-time loop.
- Do not interpret `arrival_slack_s` as permission for an unbounded
  post-crossing hold.
