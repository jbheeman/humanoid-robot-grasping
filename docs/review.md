# Aarav Implementation Review and Handoff Tasks

Reviewed commit: `b1e8ac0` (`Implement RealSense-guided G1 arm tracking`)

Status: the pure/unit test suite passes, but the implementation is **not ready for physical arm movement**. The tasks below must be completed and revalidated on the GB10 and robot before enabling `--allow-movement` or `EXECUTE=1`.

## Intended deployment topology

```text
Robot 192.168.0.213
  videohub H264 relay ───────────────► GB10 UDP 5600
  RealSense Z16 service :8767 ───────► GB10 hardware-depth fusion
  authenticated arm bridge :8766 ◄─── GB10 joint targets (execute mode only)
                                             │
GB10                                        ├─ inference/API :8000
  YOLO + RGB/depth fusion + tracking + IK   └─ viewer :8080
                                             │
                                             ▼
                                  Laptop browser on the LAN
                                  or SSH-forwarded localhost
```

The robot should run only the compressed RGB relay, RealSense depth service, and persistent disarmed arm bridge. YOLO, fusion, tracking, calibration processing, IK, and the website run on GB10. No graphical desktop is required on either machine; the operator uses a browser from a laptop.

## Review tasks

### P1 — Make the locked IK runtime executable

**Finding:** `G1RightArmIK` imports `pinocchio.casadi`, but the locked `pin==2.7.0` installation does not provide that module. This was reproduced with `uv run --group arm`; the current GB10 setup check passes because it imports only top-level `pinocchio` and `casadi`.

**Files:** `pyproject.toml`, `uv.lock`, `scripts/gb10/setup.sh`, `src/object_tracking/arm_tracking/ik_solver.py`

**Tasks:**

- Select and pin a GB10-compatible Pinocchio/CasADi installation that exposes `pinocchio.casadi`, or refactor the solver to an API available from reproducible locked packages.
- Make setup instantiate `G1RightArmIK` with the pinned URDF instead of checking imports only.
- Add a real solver test using the pinned G1 URDF, a reachable pose, an unreachable pose, joint limits, continuity, and collision rejection.
- In dry-run, seed IK from measured/current arm state rather than seven zeroes; otherwise the `0.25 rad` discontinuity gate can reject valid solutions.

**Acceptance:** clean GB10 setup constructs the solver and a replay produces valid, continuous right-arm solutions without movement.

### P1 — Complete the calibration workflow

**Finding:** `probe` outputs camera metadata only, while `solve` requires additional template fields such as workspace, waist reference, tag-to-wrist transform, and registration residuals. The runbook writes `camera-template.yaml` but later refers to `template.yaml`. `--source apriltag` only labels manually entered XYZ coordinates; no AprilTag detection or robot-pose collection is implemented.

**Files:** `src/object_tracking/arm_tracking/calibration_cli.py`, `src/object_tracking/arm_tracking/calibration.py`, `docs/ARM_TRACKING_RUNBOOK.md`

**Tasks:**

- Generate one complete template from `probe`, including explicit operator-supplied waist reference, workspace, tag-to-wrist transform, stream profiles, and registration observations.
- Implement AprilTag image detection and known wrist-pose collection, or rename the current path as manual/external correspondence import.
- Avoid opening the RealSense color stream during `probe` while `videohub_pc4` owns it; query factory profiles/extrinsics or the robot depth service instead.
- Use one filename consistently through probe, collect, solve, validate, robot startup, and GB10 startup.
- Add an end-to-end CLI test that starts from probe-like metadata and produces a hash-valid execution calibration.

**Acceptance:** at least eight solve and four held-out poses produce a calibration that passes serial/profile, registration (`<=3 px` median and `<=6 px` maximum), waist (`<=3°`), and held-out 3D residual gates.

### P1 — Verify the live ROS camera before binding an execution calibration

**Finding:** the ROS depth source reports no serial or firmware. `bind_validated_calibration` substitutes the expected YAML serial when the live serial is absent and does not compare live ROS intrinsics, allowing a different camera with matching dimensions/depth scale to be stamped with the approved calibration ID.

**Files:** `scripts/robot/depth_service.py`

**Tasks:**

- Read live serial, firmware, depth scale, stream profile, intrinsics, distortion, and RGB/depth extrinsics from ROS device-info, camera-info, and extrinsics topics/services.
- Refuse calibration binding when any required identity/profile field is unavailable or mismatched.
- Include the verified profile/hash in depth health and every streamed frame.
- Fault-inject changed serial, intrinsics, resolution, firmware, and depth scale in integration tests.

**Acceptance:** no ROS or direct-depth frame receives the execution calibration ID unless every live camera/profile field matches the validated YAML.

### P1 — Preserve perception acquisition age through the arm TTL

**Finding:** depth age is telemetry only. After fusion and potentially slow IK, the GB10 runtime assigns `source_timestamp=time.time()` immediately before the HTTP request, so stale perception can appear fresh to the robot's `250 ms` target TTL.

**Files:** `src/object_tracking/arm_tracking/runtime.py`, `src/object_tracking/arm_tracking/arm_bridge.py`

**Tasks:**

- Carry the paired RGB/depth acquisition or GB10 receipt timestamp through filtering, prediction, IK, and the arm request.
- Reject before and after IK if total target age exceeds `250 ms`.
- Keep the robot TTL based on locally measured receipt age plus an explicitly validated cross-host clock mapping, or send an age/budget that does not depend on synchronized wall clocks.
- Add artificial slow-IK, queueing, clock-jump, and network-delay tests.

**Acceptance:** a target derived from data older than `250 ms` can never be accepted, even when the outgoing HTTP request itself is new.

### P1 — Implement real controller/motion-mode arbitration

**Finding:** the Unitree adapter hard-codes `compatible_motion_mode=True` and `controller_available=True`. The process file lock detects only another copy of this bridge, not XR teleoperation or another DDS publisher.

**Files:** `src/object_tracking/arm_tracking/arm_unitree.py`, `src/object_tracking/arm_tracking/arm_bridge.py`

**Tasks:**

- Determine the supported Unitree signal/API for active motion mode and arm-controller ownership.
- Refuse arming when teleoperation or another arm publisher/controller is active.
- Continuously monitor arbitration after enable and fault/release if ownership changes.
- Add fake-SDK tests for incompatible mode and ownership loss.

**Acceptance:** a competing controller prevents enable and forces a bounded release if it appears while armed.

### P2 — Separate balance motion from commanded arm motion

**Finding:** the standing predicate requires all first 29 joint velocities, including both moving arms, to remain under `0.25 rad/s`, while the controller permits `0.50 rad/s`. A valid arm motion can trigger `standing_state_lost` and release mid-movement.

**Files:** `src/object_tracking/arm_tracking/arm_unitree.py`

**Tasks:**

- Calculate balance readiness from legs, waist, IMU attitude/angular rate, supported motion mode, and foot contact when available.
- Exclude commanded arm velocity from the standing predicate and monitor arm following error separately.
- Record each failed balance subcondition in `/state` for operator diagnosis.

**Acceptance:** bounded arm movement does not invalidate standing, while tilt, stance loss, excessive base/leg motion, and stale LowState still release safely.

### P2 — Make service transition to execute mode coherent

**Finding:** the aggregate robot launcher exits and kills all children when any child exits, but the runbook tells the operator to restart only the arm bridge with movement enabled.

**Files:** `scripts/robot/start.sh`, `docs/ARM_TRACKING_RUNBOOK.md`

**Tasks:**

- Add an explicit aggregate `dry-run`/`execute` mode, or use a supervisor with separate managed units for RGB, depth, and arm.
- In execute mode, require the validated calibration, token file, `--allow-movement`, and a deliberate operator confirmation.
- Keep startup disarmed; `/arm/enable` remains a separate authenticated action.

**Acceptance:** transitioning the arm bridge does not terminate RGB/depth, and a single service failure still stops or faults the system predictably.

## How to run after the P1 tasks are fixed

### 1. Robot

Install the RealSense ROS or `pyrealsense2` system backend, then the minimal robot environment:

```bash
uv run g1 setup robot
```

Start RGB relay, depth, and the disarmed arm bridge:

```bash
CLIENT_IP=<GB10_IP> \
ARM_TOKEN_FILE="$HOME/.config/g1-arm-token" \
CALIBRATION=/secure/runtime/g1-camera.yaml \
uv run g1 robot start
```

This is the only service group that should run on the robot for the tracking pipeline. Do not install YOLO, Torch, training data, or the GB10 IK environment there.

### 2. GB10 workstation

Install and verify the inference/IK environment:

```bash
uv run g1 setup gb10
```

Provision the ignored YOLO checkpoint, copy the validated calibration to a protected runtime path, then start dry-run processing:

```bash
ROBOT_HOST=192.168.0.213 \
CALIBRATION=/secure/runtime/g1-camera.yaml \
uv run g1 gb10 start
```

GB10 then hosts:

- `http://<GB10_IP>:8000/health` — inference and arm-tracking health
- `http://<GB10_IP>:8000/stream.mjpg` — annotated RGB stream
- `http://<GB10_IP>:8000/arm-tracking` — target/depth/IK telemetry
- `http://<GB10_IP>:8080/unitree_dual_viewer.html?single=1` — browser viewer

### 3. Laptop display

If the laptop and GB10 are on the same trusted LAN and GB10 permits inbound ports, open directly:

```text
http://<GB10_IP>:8080/unitree_dual_viewer.html?single=1
```

No display process is needed on the robot. The website is hosted on GB10 and rendered by the laptop browser.

If the ports are firewalled or the website should remain bound to localhost, use SSH forwarding:

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -L 8000:127.0.0.1:8000 \
  USER@<GB10_IP>
```

Then open:

```text
http://127.0.0.1:8080/unitree_dual_viewer.html?single=1
```

Only forward robot ports `8766`/`8767` for diagnostics when necessary. Keep the bearer token and arm endpoints off untrusted networks.

## Validation order for the next agent

1. Fix and instantiate IK on GB10.
2. Complete calibration generation and live camera identity checks.
3. Replay recorded/synthetic RGB+depth through dry-run at `>=10 Hz`.
4. Verify the website and telemetry from the laptop.
5. On supported hardware, test disarmed depth and calibration only.
6. With a spotter, exclusion zone, and physical e-stop, test controller arbitration and deadman.
7. Perform at most a `0.05 rad` right-arm smoke delta.
8. Only after every prior gate passes, attempt the 20 cm stand-off pregrasp.
