# Roadmap: tabletop plush pointing

## Scope

The G1 remains stationary. This project initially controls only the open right
arm toward a plush moving on the tabletop shown by the RGB camera. No
locomotion, torso/head motion, grasping, catching, or depth-aware reach is in
scope.

The tabletop is the correct first localization surface: it is close to the
camera, fixed, flat, well-lit, and easy to measure with a ruler. It gives a
more useful and repeatable target plane than the floor.

## Decision: RGB tabletop-plane calibration first

For the first arm-toward-plush milestone, do **not** combine the current RGB
relay with D435 depth. The two streams may be from different cameras and the
right-hand depth preview is visibly not RGB-aligned. Pairing their pixels
would be incorrect.

Instead use a planar homography:

```text
YOLO bottom-center pixel in RGB
            ↓
RGB image → tabletop (x, y) in centimetres
            ↓
taught tabletop (x, y) → bounded right-arm pointing pose
            ↓
/g1/arm/target through the guarded G1 ROS bridge
```

A homography is exactly the right math for a fixed camera observing a flat
table. It maps four or more known tabletop points to their image pixels. The
target is the **bottom-center** of the plush bounding box, approximating where
the plush contacts the table; do not use the box center.

This gets a reliable arm-toward-plush demonstration without requiring camera
height, head angle, torso origin, or raw depth registration.

## Physical setup: do this now

### 1. Keep the scene fixed

- Keep the G1 disarmed and in the same standing position.
- Do not tilt/move the camera or table after calibration.
- Keep the physical e-stop and spotter available before later arm work.
- Use the central tabletop area only; avoid table edges and the black cable
  tray, which are poor first targets.

### 2. Place four tape crosses, not lines

Use small, high-contrast `+` tape crosses, around 2–3 cm wide. Put them well
inside the usable tabletop rectangle, each clearly visible in the RGB feed.

Choose a tabletop coordinate origin at the **near-left tape cross as viewed by
the robot camera**. Use a ruler to set a rectangle, for example:

| Mark | Table coordinate |
| --- | --- |
| A — near-left | `(0 cm, 0 cm)` |
| B — near-right | `(50 cm, 0 cm)` |
| C — far-left | `(0 cm, 35 cm)` |
| D — far-right | `(50 cm, 35 cm)` |

The exact rectangle dimensions may be smaller or shifted to fit the image;
record the dimensions you actually measure. Keep a fifth cross at
`(25 cm, 17.5 cm)` as a validation mark, not as a fit point.

Put the plush in the rectangle. It will be the initial allowed target region.

### 3. Check the live RGB view

Before any arm motion, verify:

- all four corner crosses and the validation cross are sharp and visible;
- the full allowed rectangle is visible in the RGB relay;
- YOLO sees the plush reliably on the tabletop;
- the table and camera do not move between samples.

## Stage 1 — camera-to-table calibration, no robot motion

Add a GB10 calibration capture mode:

1. Click the image center of A, B, C, and D, or detect a printed fiducial at
   each mark.
2. Enter the four measured tabletop coordinates once.
3. Solve the image-to-table homography with OpenCV.
4. Project the fifth validation mark through the homography and show its error
   in centimetres and pixels.
5. Save camera stream identity, RGB resolution, the four image points, table
   points, homography, and validation result as a versioned artifact.

Accept only when the held-out center mark is within **2 cm** on the tabletop.
If it is not, replace the tape crosses with more visible markers, click more
carefully, and recalibrate. A four-corner planar mapping should be stable while
the camera and table remain fixed.

OpenCV documents planar point mapping and its perspective geometry:
[homography tutorial](https://docs.opencv.org/master/d9/dab/tutorial_homography.html).

## Stage 2 — guarded arm commissioning

Before the map is allowed to produce movement, complete
[`ARM_COMMISSIONING_RUNBOOK.md`](ARM_COMMISSIONING_RUNBOOK.md). It provides the
robot-specific home pose, confirms joint signs with tiny supervised jogs, and
validates stop/release behavior.

Do not use `scripts/robot/direct_plushie_track.py`; it bypasses this guarded
ROS command path.

## Stage 3 — teach tabletop pointing poses

After commissioning, teach five conservative pointing poses:

| Pose | Table location |
| --- | --- |
| center | `(25 cm, 17.5 cm)` |
| left | `(10 cm, 17.5 cm)` |
| right | `(40 cm, 17.5 cm)` |
| near | `(25 cm, 8 cm)` |
| far | `(25 cm, 27 cm)` |

For each location:

1. Put the plush on that tabletop coordinate.
2. Wait for a stable YOLO target and homography-projected `(x, y)`.
3. With a spotter/e-stop, use guarded commissioning controls to set an open,
   safe arm posture that points toward—not contacts—the plush.
4. Capture the **measured** right-arm joint state and table `(x, y)`.
5. Return to home before the next point.

The teaching data is the localization from tabletop `(x, y)` to a safe arm
pose. It avoids guessing camera-to-torso angle or hand-eye geometry.

## Stage 4 — dry-run pointing map

GB10 maps the tracked plush's tabletop `(x, y)` to an interpolated pose only
inside the convex hull of taught points. It shows a commanded arm ghost and
publishes no target.

Required rejections:

- target outside the tape rectangle/taught hull;
- YOLO confidence below 0.60 or less than three stable frames;
- RGB target age over 200 ms;
- table calibration mismatch, arm-state fault, or joint-envelope violation.

Validate 20 static plush placements, including held-out positions. Each
proposed posture must point in the correct direction and remain inside its
approved joint envelope. `target_sent_count` must remain zero.

## Stage 5 — limited live pointing

Enable movement only after every prior gate passes. Start stationary at the
center, then move one tabletop location at a time.

- right arm only; fixed stance; hand open;
- maximum 10 Hz updates and initially at most 1 degree per update;
- any target loss/rejection holds briefly then performs bounded release;
- the arm always stops short of the table and plush.

Success is the wrist/hand visibly moving toward the plush while it remains in
the taught tabletop region. It does not need to touch it.

## Moving plush and prediction

After static pointing works, move the plush slowly within the taped rectangle.
Log the tabletop `(x, y)` trajectory and first compare a 150 ms alpha-beta
prediction against later observed tabletop position. Only use prediction for
arm targets after it improves or matches the baseline on held-out runs.

## Later 3D upgrade

For standoff reach, grasp, or interception, add RGB-aligned depth and a
validated camera-to-torso transform. The D435 depth stream must never be
indexed with pixels from a different RGB relay without an explicit inter-camera
calibration.
