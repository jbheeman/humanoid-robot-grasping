# Roadmap: top-down D435 plush pointing

## Correct physical model

The D435I looks down at the floor from the upper G1 body. The useful first
calibration object is therefore the **floor**, not a vertical board.

For a plush on the floor, the system needs:

1. RGB pixel `(u, v)` from YOLO;
2. D435 depth at that pixel, giving a 3D point in the camera optical frame;
3. one fixed transform from camera optical coordinates to the robot torso
   frame;
4. a conservative arm target/pointing posture derived from that torso point.

The RealSense supplies the per-frame range. A ruler is used only to establish
the camera-to-robot geometry once; it is not used every time the plush moves.

## The simplest calibration that is still meaningful

Use a few **floor marks** measured relative to the stationary robot, then fit
camera-to-torso geometry from their observed RGB/depth positions.

One measurement is not enough: camera height, forward/back and lateral offset,
pitch, roll, and yaw all affect the mapping. The floor plane from depth gives
height/pitch/roll strongly; two or more known floor points establish the
remaining in-plane position/yaw. In practice, use six marks to fit and four
separate marks to validate. This is far easier than measuring a moving plush
in arbitrary 3D space.

## Physical setup

### 1. Keep the robot fixed

- Stand the G1 in its normal stance on a flat floor.
- Keep it disarmed. Do not move the torso/head or relocate the camera after
  calibration.
- Mark the feet with tape so the same standing position can be restored.
- Use a spotter and e-stop whenever arm commissioning begins later.

### 2. Choose the floor coordinate convention

Use the horizontal floor plane as the initial working surface:

- `x`: forward from the torso/centerline;
- `y`: robot-left;
- `z = 0`: floor.

The exact torso origin can be an approximate point centered between the hip
axes, vertically projected to the floor. What matters is that every mark uses
the same origin and axes. For the first arm-toward-plush demo, centimetre-level
absolute perfection is not required; held-out validation decides whether the
fit is good enough.

### 3. Make ten tape marks on the visible floor

Put small pieces of non-reflective tape or paper dots where the camera can see
them. Write each name and measured `(x, y, 0)` in metres on a note.

Use these positions as a starting pattern, adjusting to your visible floor:

| Name | Suggested torso-floor coordinate (m) |
| --- | --- |
| center | `(0.55, 0.00, 0.00)` |
| left | `(0.55, 0.25, 0.00)` |
| right | `(0.55, -0.25, 0.00)` |
| near | `(0.35, 0.00, 0.00)` |
| far | `(0.80, 0.00, 0.00)` |
| high_left | `(0.70, 0.30, 0.00)` |
| high_right | `(0.70, -0.30, 0.00)` |
| low_left | `(0.40, 0.20, 0.00)` |
| low_right | `(0.40, -0.20, 0.00)` |
| off_axis | `(0.65, 0.12, 0.00)` |

These are **not** required to be exact numbers. Measure the actual distance
from your chosen origin/centerline with a tape measure and enter those actual
numbers. The marks must cover the floor area where the plush will initially
move.

## Calibration capture

The existing no-actuation tuner is suitable for this setup. Place the plush on
each floor mark, draw its box, and enter the mark's measured coordinates:

```bash
# On G1, with the robot disarmed
cd ~/humanoid-robot-grasping
python3 scripts/robot/localization_tuner.py runs/localization/floor_poses.json
```

For example, if the plush is on the measured `left` mark:

```text
name torso_x torso_y torso_z metres:
left 0.55 0.25 0.00
```

The tuner captures the real D435 intrinsics, aligned RGB/depth image, selected
plush box, and median depth. It never commands the robot.

Fit with six marks and keep four unseen marks for validation:

```bash
# On GB10
cd ~/Documents/project
uv run g1 calibrate localization \
  runs/localization/floor_poses.json \
  --fit center left right near far high_left \
  --validation high_right low_left low_right off_axis \
  --output runs/localization/floor_report.json
```

Pass only when the report has median error ≤5 cm, p95 ≤8 cm, and no left/right
or up/down inversion. If it fails, remeasure the tape marks, ensure the plush
is centered on each mark, and keep the robot/camera fixed.

## First arm behaviour

After the floor transform validates, do **not** start with grasping or future
prediction. First run a dry simulation of a pointing target: YOLO + aligned
depth → camera XYZ → torso XYZ → conservative arm pointing posture/IK. The UI
must show the measured floor plush point, proposed arm pose, and every
rejection, while sending no arm target.

Only after dry-run checks and existing arm commissioning pass, enable a fixed
stance, open-hand, right-arm-only mode. Keep the plush in the calibrated floor
area. The arm should move toward the target direction but stop short of the
floor/plush by a conservative standoff distance.

## Prediction, later

Once measured torso XYZ is valid, log moving plush trajectories. Score a
150 ms predicted position against the later observed position. Use the
alpha-beta fallback unless the learned model wins on held-out real trajectories.
Prediction remains dry-run before it affects arm targets.

## Constraints that stay in force

- no locomotion, head/torso rotation, hand closure, catch, or grasp;
- no targets outside the calibrated floor region or arm workspace;
- stale RGB/depth, lost plush, low confidence, failed IK, or bridge fault:
  hold/release, never continue;
- GB10 computes intent; the G1 ROS bridge remains the only hardware authority.

## References

The RealSense SDK documents depth-to-color alignment and use of stream
intrinsics for projection/deprojection:
[RealSense alignment example](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py) and
[projection/deprojection overview](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/).
