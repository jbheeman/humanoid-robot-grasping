# RealSense-Guided G1 Arm Tracking

## Summary

- Keep the robot's existing `videohub_pc4` H264 multicast relay for RGB video.
- Use Intel RealSense Z16 hardware depth as the only depth source; do not add learned depth estimation.
- Run plushie YOLO inference, RGB/depth fusion, 3D tracking, calibration, and IK on GB10.
- Run SDK2 locally on robot `192.168.0.213` through a persistent authenticated REST bridge.
- Initially control only the 29-DOF G1 right arm, with the waist locked and no hand closure. Move to a safe pregrasp pose approximately 20 cm from the plushie.
- Preserve the current 30 FPS RGB fast path, tolerate a 15 FPS source, run arm target generation at 10-15 Hz, and interpolate commands locally on the robot at 250 Hz.

## Implementation

### Robot Services

- Preserve compressed H264 multicast forwarding; do not decode RGB or run YOLO on the robot.
- Add a RealSense depth service that prefers an existing ROS aligned-depth topic and falls back to direct `librealsense` depth-only capture using factory intrinsics and extrinsics.
- Use Z16 depth at the best supported 30 FPS mode, retain only the newest frame, and transmit at 15 FPS to bound bandwidth while supporting 10-15 Hz target generation.
- Apply RealSense spatial, temporal, validity, and outlier filtering without replacing measurements with learned estimates.
- Fail closed if `videohub_pc4` prevents depth access; never silently substitute monocular depth.
- Expose depth through `GET /health`, `GET /depth/calibration`, and `WS /depth/stream` with timestamped, zstd-compressed Z16 frames.
- Add a dedicated arm bridge on port `8766`, separate from the subprocess-based locomotion bridge.
- Base its persistent controller on Unitree's official `rt/arm_sdk` approach and arm-weight ramping.
- Run the robot control loop at 250 Hz, command all 14 arm joints, hold the left arm at its measured pose, and interpolate the seven right-arm joints.
- Require `--allow-movement`, a bearer token, explicit `/arm/enable`, balanced standing state, and valid calibration before accepting targets.
- Expose `GET /health`, `GET /state`, `POST /arm/enable`, `POST /arm/target`, and `POST /arm/stop`.
- Enforce official URDF joint limits, conservative velocity/acceleration limits, a 250 ms command TTL, and a 500 ms deadman. Target loss, network loss, stale commands, or controller faults must hold briefly and ramp arm weight to zero.
- Add one robot command that starts the RGB relay, depth service, and disarmed arm bridge without installing training or CUDA dependencies.

### GB10 Perception and Control

- Build the 3D arm-tracking pipeline around the committed YOLO server and tracker; the previously referenced uncommitted prototype is not present in the repository.
- Continue using `models/plushie_detector/yolov8n_plushie_mvp/weights/best.pt`.
- Pair each decoded RGB frame with the nearest depth frame, rejecting pairs more than 100 ms apart.
- Estimate object depth using valid-depth clustering and a robust median from the central detection region.
- Deproject the pixel and Z16 depth using RealSense intrinsics, then transform the point into the locked-waist G1 base frame.
- Maintain a filtered 3D position and velocity estimate and predict approximately 150 ms ahead.
- Extract the support plane from the hardware depth map and require at least 10 cm wrist clearance.
- Generate a right-wrist target 20 cm back from the object along the shoulder-to-object approach vector with a fixed neutral palm orientation.
- Adapt Unitree's official G1 29-DOF Pinocchio/CasADi IK and locked-waist URDF with preserved license notices and pinned source revisions.
- Reject unreachable, self-colliding, workspace-violating, depth-uncertain, or discontinuous IK results.
- Keep dry-run as the default. Require `--execute`, the bearer token, valid calibration, and a healthy arm bridge for movement.
- Add an isolated `arm` dependency group for GB10 IK and calibration dependencies. Robot setup must continue using only minimal vision, depth, and loco dependencies.

### Calibration and Viewer

- Add calibration commands for `probe`, `collect`, `solve`, `validate`, and bounded XYZ/RPY tuning.
- Calibrate the rigid camera-to-torso transform using at least eight known right-arm poses and an AprilTag attached to the wrist, with manual point selection as a fallback.
- Store camera serial, intrinsics, depth scale, RGB/depth extrinsics, camera-to-torso transform, waist reference, residuals, timestamp, and calibration ID in YAML.
- Refuse execution if the camera serial changes, the waist differs by more than 3 degrees from calibration, or 3D validation RMSE exceeds 25 mm.
- Extend the GB10 viewer on port `8080` with RGB detections, a hardware-depth colormap, target XYZ, depth age and validity, predicted position, IK status, arm state, and dry-run or armed status.
- Document the robot, GB10, and MacBook startup and port-forwarding workflow.

## Tests and Acceptance

- Unit-test target selection, Z16 decoding, RGB/depth pairing, robust ROI depth, deprojection, transforms, 3D prediction, workspace checks, and target-loss behavior.
- Test IK against the official 29-DOF URDF for joint limits, continuity, unreachable targets, and left-arm preservation.
- Integration-test depth transport and the arm REST API with recorded or synthetic depth frames, authentication failures, stale sequences, network disconnects, and deadman release.
- Replay a recorded RGB/depth sequence through the full GB10 pipeline without robot movement.
- Stage hardware verification through depth-only validation, calibration, a maximum `0.05 rad` arm smoke movement, dry-run pregrasp, and finally an executed 20 cm stand-off reach.
- Acceptance requires at least 10 Hz fresh target updates from the 15 FPS feed, depth pairing within 100 ms, automatic stop within the deadman interval, and wrist placement within 5 cm of the planned pregrasp target.

## Delivery

- Preserve and rework the current prototype files.
- Update setup scripts, lockfile, README, calibration runbook, REST schemas, and MacBook port-forwarding instructions.
- Run tests and static checks, confirm no unintended CUDA or training installation on the robot, then commit and push completed work to `origin/aarav`.

## Upstream References

- Unitree XR Teleoperate: <https://github.com/unitreerobotics/xr_teleoperate>, pinned during design at `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`.
- Unitree ROS descriptions: <https://github.com/unitreerobotics/unitree_ros>, pinned during design at `d96d8f63ae17a7108d4f7229c00ef875ba7129c9`.
- RealSense ROS: <https://github.com/realsenseai/realsense-ros>.
