# Plush trajectory prediction and arm interception roadmap

## Goal and boundary

The first hardware goal is not a catch or a grasp. It is a supervised,
arm-only **block posture**: predict where a moving plush will cross a fixed
plane in front of a stationary G1, then move the right arm slowly into a
bounded posture at that plane.

The robot must not rotate, walk, reach toward people, close its hand, or make
contact-dependent decisions in this roadmap. Those are separate later
projects. Every stage below begins with no arm commands and advances only when
its acceptance criteria are recorded.

## System ownership

| Host | Responsibility | Must not do |
| --- | --- | --- |
| G1 (`192.168.0.213`) | D435I depth publication, robot ROS bridge, final arm safety gate | YOLO inference or browser UI |
| GB10 (`192.168.0.66`) | RGB decoding, YOLO, tracking, 3D state estimation, prediction, IK, browser UI | Direct Unitree SDK control |
| Operator | Fixture placement, e-stop, stop/go decision, ground-truth records | Enable movement from an unverified prediction |

The GB10 publishes only bounded intent through `/g1/arm/target`. The G1
bridge is the only process allowed to communicate with robot hardware and may
always reject a target.

## Stage 0 — transport and viewer baseline

### Software state

- Start the G1 bridge disarmed.
- Start GB10 in `--dry-run` mode.
- Confirm GB10 discovers `/g1/arm/state`, `/g1/depth`, and
  `/g1/commissioning/state` on ROS domain 42.
- Confirm the GB10 UI shows annotated RGB and raw D435 depth.

### Acceptance criteria

- RGB and depth remain live for five minutes.
- YOLO provides a stable plush track.
- No arm target is sent (`target_sent_count = 0`).
- The UI may show raw/factory depth as blocked; that is expected at this
  stage.

## Stage 1 — 2D future-position proof, no arm movement

### Purpose

Prove that the system can predict a plush's future **pixel** position while it
moves through the camera image. This validates detector stability, timestamp
handling, tracker velocity, and processing latency before 3D geometry is
introduced.

### Required implementation

- Draw the current tracked center, a recent pixel trail, and a predicted
  center at 150 ms, 300 ms, and 500 ms ahead in the annotated RGB view.
- Log the prediction timestamp, horizon, measured later center, and pixel
  error for every due prediction.
- Compare the learned forecaster with the alpha-beta fallback; use the
  fallback whenever history is too short or the learned prediction is
  implausible.

### Physical procedure

1. Clear the arm workspace and keep robot motion disabled.
2. Put the plush on a simple repeatable path: slide it on a table, swing it on
   a string, or move it by hand horizontally. Do not throw it yet.
3. Keep the full path in the RGB view for 20–30 seconds at a time.
4. Record at least ten runs: slow left/right, faster left/right, and diagonal
   motion.

### Acceptance criteria

- Prediction overlay is visible and timestamped.
- The report contains at least 100 scored predictions per tested horizon.
- Prediction is compared with a constant-velocity alpha-beta baseline, not
  merely displayed.
- No target is sent to the arm bridge.

## Stage 2 — fixed-board camera-to-torso calibration

### Why this is required

Interception is a 3D problem. Pixel coordinates cannot identify a safe arm
position or account for depth. The current D435 raw Z16 stream is not
registered to the RGB image, so it cannot produce a valid torso-frame plush
trajectory until a camera-to-torso transform and depth registration are
validated.

### Use a board/jig, not freehand point measurements

Build or print a rigid flat board with at least six clearly marked points,
preferably AprilTags or high-contrast circles at known board coordinates.
Mount or hold it in a repeatable fixture in front of the stationary robot.
The board needs a documented pose relative to the robot torso only once. Its
corner/marker coordinates provide the repeated correspondences automatically.

### Physical setup

1. Keep the robot disarmed, standing still, with a spotter and e-stop.
2. Choose a torso-frame convention and write it on the fixture record:
   `+x forward`, `+y robot-left`, `+z upward`; units are metres.
3. Place the board roughly 0.4–1.2 m in front of the chest, fully visible in
   the D435 RGB image and not reflective.
4. Use a tape measure and level once to record the board origin and axes
   relative to the chosen torso origin. Photograph the setup.
5. Capture at least ten board observations across the useful camera field:
   center, left, right, high, low, far, near, upper-left, lower-right, and
   off-axis. Move the board, not the robot.

### Required software work

- Add a board capture mode that detects marker corners and stores their known
  board coordinates, RGB/depth samples, camera intrinsics, and timestamps.
- Fit optical-to-torso transform using six named observations and reserve four
  for held-out validation.
- Add a promotion command that converts only a passed localization report into
  the immutable runtime calibration artifact consumed by GB10 and the G1
  bridge.

### Acceptance criteria

- Median held-out localization error ≤ 5 cm.
- p95 held-out localization error ≤ 8 cm.
- Left/right and up/down direction checks pass.
- Registered depth shape and RGB shape agree.
- Calibration artifact has a UUID, hash, camera serial, profile, transform,
  and validation evidence.

## Stage 3 — 3D prediction shadow mode

### Software state

Run GB10 with the validated calibration artifact and `--dry-run`. The runtime
may deproject the tracked plush, transform it to torso coordinates, maintain a
3D velocity estimate, and run the learned trajectory model. It still must not
send arm targets.

### Required telemetry

- measured torso XYZ and velocity;
- 150/300/500 ms predicted XYZ;
- predictor source, checkpoint, and fallback reason;
- online later-observation error by horizon;
- RGB/depth skew, depth age, confidence, and target age;
- candidate block-plane point and IK result.

### Physical procedure

1. Repeat the Stage 1 plush paths at slow and moderate speed.
2. Keep the plush inside the calibrated board volume.
3. Record at least 20 trajectories with varied directions and depths.
4. Inspect the predicted 3D point against the later measured point before
   reviewing any IK result.

### Acceptance criteria

- At least 100 scored real 3D predictions at the intended horizon.
- Learned model is retained only if it beats or matches the alpha-beta
  fallback on held-out real trajectories.
- No stale depth, stale RGB, calibration mismatch, or invalid IK becomes an
  arm command.
- `target_sent_count = 0` throughout shadow mode.

## Stage 4 — IK shadow block plane

### Purpose

Convert a future plush point into a conservative arm posture without commanding
hardware. The intercept target is constrained to a fixed, pre-approved plane;
it is not a general reach target.

### Required constraints

- fixed robot base and torso;
- right arm only;
- fixed shoulder/elbow workspace and joint limits;
- target update rate, joint slew, acceleration, and following-error limits;
- prediction/target freshness watchdog;
- zero-weight release on lost target, stale data, bridge loss, or operator
  stop;
- no hand close and no locomotion.

### Acceptance criteria

- The predicted point is within the approved block plane and workspace.
- IK is valid for repeated trajectories without limit or clearance violations.
- The visualizer shows measured arm state, commanded ghost, predicted plush,
  and candidate block posture.
- No hardware arm target is sent.

## Stage 5 — supervised low-speed arm block

This is the first stage that permits arm movement. It requires separate arm
commissioning, a passed calibration artifact, a cleared workspace, a trained
spotter, and a physical e-stop.

Start with a stationary plush and a fixed target posture. Progress to slow,
bounded lateral plush motion only after the observed arm motion matches the
commanded ghost. The arm must stop or release on every rejected target.

### Acceptance criteria before expanding scope

- Independent observation confirms arm movement direction and magnitude.
- The arm reaches the approved block posture without overshoot or safety
  rejection.
- Prediction error and end-effector error are recorded for every trial.
- At least 20 supervised successful trials complete with no safety violation.

## Explicitly deferred

- catching or grasping a moving plush;
- hand closure;
- body rotation or locomotion;
- tracking a plush outside the camera field;
- collision/contact reaction;
- autonomous activation without an operator.

## What the operator does next

Begin Stage 1 today: keep the G1 disarmed, start the GB10 server in dry-run,
and collect repeatable moving-plush video. In parallel, prepare the fixed
board fixture for Stage 2. Do not use the current factory/raw depth status to
activate 3D prediction or arm movement.
