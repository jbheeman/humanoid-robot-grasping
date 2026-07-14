# Roadmap: point the G1 arm toward a plush

## Decision

The first milestone is **image-plane pointing**: with a stationary G1 and a
plush constrained to a marked plane in front of the camera, move the right arm
toward the plush's detected image position. This is deliberately not a 3D
reach, grasp, catch, or intercept.

Use a **taught 2D pointing map**. The operator teaches five to nine safe arm
postures against known cells on a fixed board. GB10 maps a fresh YOLO plush
center to an interpolated, bounded right-arm pose and sends it through the
existing guarded ROS bridge.

This is the highest-value first route because it needs no hand-measured torso
XYZ, does not misuse the current unregistered raw Z16 depth, and uses the
existing GB10 YOLO → `/g1/arm/target` → G1 safety bridge path.

## What is deferred

- 3D reach, distance-aware standoff, grasping, catching, and contact;
- locomotion, torso/head rotation, or tracking outside the camera view;
- learned 3D trajectory prediction in the command path.

The current trajectory model stays observation-only until RGB-aligned depth
and camera-to-torso geometry are validated.

## Why this instead of full localization now?

| Option | Decision | Reason |
| --- | --- | --- |
| Manual plush torso XYZ samples | Later | Operator-heavy and easy to measure inconsistently. |
| Full wrist-tag hand-eye calibration | Later 3D upgrade | Accurate, but needs multi-pose supervised arm collection. |
| 2D visual-servo Jacobian | Later upgrade | Useful continuous correction, but needs wrist-tag identification. |
| **Taught 2D pointing map** | **Now** | Directly maps a plush image location to pre-approved safe poses. |

Use a printed AprilTag grid as a repeatable board reference if available. It
does not require torso measurements in this stage. AprilTag pose estimation
requires the true tag edge length and camera intrinsics; OpenCV PnP estimates
pose from known object points and their image projections. Sources:
[AprilTag ROS documentation](https://docs.ros.org/en/rolling/p/apriltag/) and
[OpenCV PnP](https://docs.opencv.org/master/d5/d1f/calib3d_solvePnP.html).

## Safety boundary

- The G1 stays standing and fixed; right arm only; hand open.
- GB10 provides intent only. G1 remains the only Unitree SDK/hardware owner.
- Never run `scripts/robot/direct_plushie_track.py` for this workflow.
- A lost target, stale RGB, confidence drop, outside-board pixel, bridge fault,
  state mismatch, or operator stop holds/releases the arm; no extrapolation.
- Initial commands are limited to approved joints and a small envelope around
  taught poses, with the existing bridge's limits and stop/release behavior.

## Physical setup: do this first

### Materials

- Rigid matte board, roughly 60–90 cm wide and 45–70 cm high.
- A clearly drawn 3×3 grid, with cells at least 15 cm apart.
- Plush, e-stop, and spotter.
- Optional: AprilTags in the board corners and a small flat wrist marker.

### Setup procedure

1. Keep the G1 **disarmed** and standing still.
2. Place the board about 0.7 m in front of the chest D435I. Adjust it until
   every cell, the plush, and the right wrist/hand are visible in GB10 RGB.
3. Tape or clamp the board, then mark its feet. It must not move during
   teaching or tracking.
4. Put the plush at the center cell. Confirm YOLO gives a stable box for ten
   seconds.
5. If the hand is hard to see, rigidly attach a small matte high-contrast
   marker or AprilTag to the outside of the wrist/hand.

### Before any arm motion

- YOLO confidence is at least 0.60 for three consecutive frames at each of
  `center`, `left`, `right`, `up`, and `down`.
- RGB/timestamps are fresh in the GB10 viewer.
- The robot remains disarmed and no target is published.

## Stage 1 — guarded arm commissioning

Complete [`ARM_COMMISSIONING_RUNBOOK.md`](ARM_COMMISSIONING_RUNBOOK.md) on the
G1 before teaching poses. This records the measured right-arm home pose and
performs existing tiny supervised sign checks. Do not use guessed joint signs.

Advance only when:

- the right-arm home profile is saved for this robot;
- each proposed pointing joint passes a supervised sign check;
- `/g1/arm/state` is healthy;
- stop/release behavior has been observed successfully.

## Stage 2 — teach five safe pointing poses

Start with `center`, `left`, `right`, `up`, and `down`; expand to all nine
cells after this works. For each point:

1. Put the plush at the cell center and wait for a stable YOLO lock.
2. With spotter/e-stop, use guarded commissioning controls to slowly position
   the open right arm toward the plush. Keep a generous board clearance.
3. Capture: board ID, plush pixel center, confidence, measured right-arm joint
   state, optional wrist-marker pixel, and timestamp.
4. Return to home before teaching the next point.

Reject a sample unless board and target are stable, state is fresh, and only
approved joints differ from home.

## Stage 3 — dry-run pointing map

GB10 selects the tracked plush center, then performs bounded interpolation
between taught poses. It renders a commanded ghost but publishes **no** arm
target.

Required rules:

- interpolate only within the convex hull of taught image points;
- reject image edges and every target outside the board;
- use a pixel deadband to suppress detection jitter;
- clamp all joints to taught-pose envelopes and bridge limits;
- log desired pose, nearest cells, confidence, target age, and rejection.

Required implementation surfaces:

- `src/object_tracking/arm_tracking/pointing_map.py` — schema, interpolation,
  hull rejection, and allowed-joint envelope;
- `src/object_tracking/arm_tracking/pointing_runtime.py` — stable-target,
  freshness, deadband, dry-run telemetry;
- `src/object_tracking/yolo_stream_server.py` — runtime status;
- `scripts/gb10/web/unitree_dual_viewer.html` — board, plush/wrist markers,
  selected cells, desired pose, and rejection reason.

Tests must cover anchors, interpolation, hull/stale/lost-target rejection,
allowed joint mask, and proof that dry-run never publishes.

### Dry-run acceptance

- 20 static placements, including held-out cells.
- Desired pose changes in the correct direction every time.
- No desired joint is outside its envelope.
- Zero arm targets are published.

## Stage 4 — limited live pointing

After all prior gates pass, use the G1 movement gate with spotter and e-stop.
Begin center cell only, then neighboring cells.

- 10 Hz maximum target update rate;
- no more than 1 degree per update until observed behavior is reviewed;
- confidence ≥0.60, 3–6 stable frames, target age <200 ms;
- hand open; fixed stance; right arm only;
- lost target/rejection: hold briefly, then bounded release home.

Success means the wrist/hand moves toward the plush and settles inside the
agreed image tolerance. It does not need to touch the plush.

## Stage 5 — moving plush and later 3D

Once static pointing works, move the plush slowly within the board plane. Add
150 ms 2D prediction only after its scored error matches or beats the
alpha-beta baseline on held-out runs. Keep every predicted pixel inside the
taught board hull.

For later 3D prediction/interception, replace the taught map with RGB-aligned
depth and a validated camera-to-torso transform. A wrist-mounted AprilTag grid
paired with measured arm FK poses is the preferred no-manual-XYZ calibration.
Depth must be aligned to RGB before a YOLO box uses it; see the
[RealSense alignment example](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py).

## Next operator action

Build/place the board and confirm the GB10 viewer can see the plush in the
five starting cells plus the right wrist/hand. Do not move the arm yet. Then
complete guarded commissioning before teaching the first pose.
