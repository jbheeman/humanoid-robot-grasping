# Plan: Real-time bunny interception implementation tonight

**Generated:** 2026-07-26  
**Implementation branch:** `aarav-vla`  
**Estimated complexity:** High  
**P0 implementation target:** 2 focused hours  
**Automated verification and review budget:** 2 hours  
**Integration/debugging reserve:** 2 hours; may absorb P0 implementation overrun  
**Physical validation:** Tomorrow, using the completed dry-run and execute paths

## Outcome

Tonight's deliverable is a complete, disabled-by-default interception path that
can be configured and exercised tomorrow without writing Python:

```text
exact RGB frame processed by YOLO
  -> timestamped 3D position/velocity estimate
  -> fixed-plane crossing prediction
  -> deadline and empirical reachability gates
  -> fixed-orientation, collision-checked local IK
  -> preview
  -> commit one intercept target
  -> move and hold while observations remain fresh
```

The existing continuous tracking mode must remain unchanged when no intercept
configuration is supplied.

## Fixed assumptions

- The G1 remains stationary and uses only the right arm.
- The bunny follows one repeatable tabletop lane.
- The task is palm blocking/contact, not grasping.
- A full-pose, collision-checked ready pose establishes palm orientation before
  interception. The real-time loop preserves that orientation and performs
  translation-only local IK.
- The existing 250 ms target freshness limit, deadman, joint limits, slew and
  acceleration limits, standing checks, following-error check, collision
  checks, and bounded release remain enabled.
- Learned trajectory prediction is not required. Alpha-beta prediction is the
  default until recorded evidence shows another predictor is better.
- No new robotics framework or runtime dependency is introduced tonight.
- Hardware execution cannot be declared validated tonight. Execution remains
  behind the existing robot-side permission and explicit GB10 `--execute`
  gates.

## Prerequisites for tomorrow

- Fine-tuned YOLO engine on the GB10.
- Validated camera-to-torso/table calibration.
- Robot-specific commissioned home profile.
- Measured lane and intercept plane in the same torso frame as the calibration.
- Physically validated ready pose and palm-facing direction.
- E-stop, spotter, cleared workspace, repeatable bunny motion, and independent
  speed measurement.

The code must fail closed when any prerequisite artifact is absent, malformed,
or bound to a different calibration.

## Implementation sequence and timebox

| Window | Increment | Result |
| --- | --- | --- |
| 0:00-0:30 | Exact processed-result bundle | Current detections carry the time, ID, and shape of the frame YOLO actually processed |
| 0:30-1:00 | Config and pure intercept controller | Preview/commit/expiry decisions are deterministic and unit-testable |
| 1:00-1:45 | Runtime and CLI integration | Dry-run can form an intercept candidate; execute re-solves safe incremental edges |
| 1:45-2:00 | Essential telemetry and runbook | Health explains readiness/rejection; tomorrow's commands are documented |

If a task overruns, preserve this priority order:

1. Correct timing and fail-closed behavior.
2. Dry-run intercept preview.
3. Safe commit/hold execution.
4. Telemetry.
5. UI polish.

### Two-hour P0 cut line

The reviewer estimated the full plan at five to eight coding hours. The
two-hour implementation target therefore includes only:

1. one atomic processed-result bundle and current-detection/stale-track rules;
2. strict fresh measured-arm extraction for interception;
3. versioned config loading and disabled-by-default CLI plumbing;
4. a pure `PREVIEW -> COMMITTED -> EXPIRED/HOLD` machine with injected time;
5. a runtime intercept branch that latches a Cartesian point and repeatedly
   validates only the next local IK edge;
6. essential readiness telemetry, tests, and the morning runbook.

Rolling percentile aggregation, UI redesign, predictor comparisons, global path
proof, automatic retreat, and any probabilistic confidence score are P1 and
must not displace P0 correctness.

## Sprint 1: Make perception timing truthful

**Goal:** Every track and command refers to the exact detector frame that
produced it.

**Demo/validation:** Artificially publish raw frames faster than inference.
Verify the track snapshot retains the processed frame's ID and timestamp rather
than adopting the newest raw frame's timestamp.

### Task 1.1: Capture immutable detector-frame provenance

- **Locations:**
  - `src/object_tracking/yolo_stream_server.py`
  - `src/object_tracking/simple_tracker.py`
- **Description:**
  - When `_inference_loop` copies a raw frame, create one atomic processed
    result bundle containing frame ID, monotonic receipt timestamp, processed
    image shape, monotonic inference start/completion, detections, and tracks.
  - Publish the bundle atomically only after inference/tracking completes.
  - Treat only tracks updated by the current detection batch
    (`missed_updates == 0`) as current interception observations. An unmatched
    track retained by `SimpleTracker` must retain its own last-seen time and
    must never inherit the new batch timestamp.
  - Continue exposing wall-clock timestamps only for human-facing records.
    Filtering, freshness, and tracking use monotonic time exclusively.
- **Perceived complexity:** 4/10
- **Dependencies:** None
- **Acceptance criteria:**
  - The track batch frame ID equals the copied inference frame ID.
  - Snapshot shape comes from the processed bundle, not the newest raw frame.
  - A newer raw frame cannot make an older completed inference appear fresh.
  - An unmatched retained track cannot be selected as a fresh observation.
  - Tracker deltas never mix `time.time()` with monotonic timestamps.
  - Out-of-order or duplicate processed frame IDs are ignored.
- **Validation:**
  - Unit test with frame N processed while frames N+1 through N+5 arrive.
  - Unit test with an intentionally delayed inference completion.
  - Unit test with a retained unmatched track and a current empty batch.

### Task 1.2: Pair depth with the detector frame

- **Locations:**
  - `src/object_tracking/yolo_stream_server.py`
  - `src/object_tracking/arm_tracking/runtime.py`
- **Description:**
  - Change `arm_tracking_snapshot()` to return the processed detector-frame
    timestamp and frame ID.
  - Compute RGB/depth skew and target age from that timestamp.
  - Include detector frame ID and depth sequence in runtime status.
- **Perceived complexity:** 3/10
- **Dependencies:** Task 1.1
- **Acceptance criteria:**
  - Pair skew no longer uses `state.raw_frame_received_monotonic`.
  - Existing stale-pair and target-TTL gates are not relaxed.
  - Runtime rejects a detector frame older than the previous filter update.
- **Validation:**
  - Extend `tests/test_arm_tracking_runtime.py` with old-track/new-raw-frame and
    out-of-order cases.

### Task 1.3: Add raw segment-level latency telemetry

- **Locations:**
  - `src/object_tracking/yolo_stream_server.py`
  - `src/object_tracking/arm_tracking/runtime.py`
- **Description:** Record, where observable:
  - detector-frame receipt to inference start;
  - inference duration;
  - inference completion to depth pairing;
  - localization/filter duration;
  - planning duration;
  - IK duration;
  - target publication age;
  - existing robot acceptance/arm-loop fields alongside the new values.
- **Perceived complexity:** 5/10
- **Dependencies:** Tasks 1.1 and 1.2
- **Acceptance criteria:**
  - P0: raw segment values are present in health/status and every research
    sample.
  - P1/stretch: extend `src/object_tracking/tuning_cli.py` to report sample
    count, mean, p50, p95, and p99 and add any missing robot-acceptance fields.
  - Clock domain is named for every timestamp.
  - Cross-host capture latency is not claimed unless clocks are synchronized;
    target pipeline age and robot acceptance provide the operational freshness
    bound.
  - No sensitive payloads or image data are added to timing logs.
- **Validation:**
  - Unit tests for raw duration calculations, clock-domain labels, and missing
    segment values.
  - Percentile-output tests are P1/stretch with `tuning_cli.py`.
  - Existing research-session JSONL remains backward-compatible.

## Sprint 2: Add a pure, testable intercept decision controller

**Goal:** Separate interception decisions from ROS, YOLO, IK, and hardware so
all state transitions can be tested offline.

**Demo/validation:** Feed a synthetic straight-line track to the controller and
observe `ACQUIRING -> PREVIEW -> COMMITTED -> EXPIRED`. Feed stale, reversing,
unreachable, or noisy tracks and observe `HOLD` with a specific reason.

### Task 2.1: Add validated live-intercept configuration

- **Locations:**
  - New `src/object_tracking/arm_tracking/interception.py`
  - New `configs/intercept/demo-lane.example.yaml`
  - New tests in `tests/test_interception_runtime.py`
- **Description:** Define a configuration containing:
  - schema version and calibration ID;
  - torso-frame intercept plane point and normal;
  - permitted crossing segment/bounds;
  - wanted detector classes and minimum detector confidence;
  - validated ready-pose right-arm joints and allowed deviation;
  - physically validated lane-facing palm normal as profile evidence;
  - minimum consecutive track samples;
  - maximum estimator residual;
  - commit horizon and bounded post-crossing hold;
  - conservative compute, command, Cartesian speed, acceleration, and settle
    bounds used by `InterceptConfig`;
  - minimum deadline slack;
  - revalidation tolerance for a committed crossing;
  - explicit `validated_for_execution` flag.
- **Perceived complexity:** 4/10
- **Dependencies:** None
- **Acceptance criteria:**
  - Non-finite, non-unit, geometrically invalid, or incomplete configuration is
    rejected.
  - Commit/hold windows cannot be configured to bypass the arm target TTL.
  - Calibration mismatch is rejected.
  - Example configuration contains placeholders and cannot accidentally enable
    execution.
  - `--execute` rejects a profile not marked and validated for execution.
- **Validation:**
  - Schema/config unit tests for every rejection.

### Task 2.2: Implement the intercept state machine

- **Location:** `src/object_tracking/arm_tracking/interception.py`
- **Description:**
  - Wrap existing `plan_plane_intercept()`.
  - Track consecutive observations and estimator residual evidence.
  - Produce explicit states:
    `ACQUIRING`, `PREVIEW`, `COMMITTED`, `EXPIRED`, and `HOLD`. Additional UI
    labels must not complicate the safety state machine.
  - Latch one Cartesian target at commit.
  - Permit fresh, consistent observations to revalidate the evidence timestamp
    without moving the latched target.
  - Never permit reversal or target chasing after commit.
  - Fresh current detections may reconfirm that their newly predicted crossing
    stays within a configured tolerance of the latched Cartesian target. That
    confirmation may refresh command provenance while the target stays fixed.
  - Permit occlusion hold only until both the separately configured
    post-crossing window and the existing perception TTL allow it; never
    manufacture a fresh timestamp.
  - Treat `InterceptPlan.hold_time_s` as arrival/deadline slack in the wrapper,
    not as permission to hold after crossing.
  - Reset on track switch, current-detection miss, non-monotonic frame, target
    passed, depth loss, arm-state loss, or config/calibration change. Terminal
    expiry requires the existing stop/release and explicit re-enable/reset; no
    automatic retreat is added tonight.
- **Perceived complexity:** 6/10
- **Dependencies:** Task 2.1
- **Acceptance criteria:**
  - No crossing, too-slow motion, passed crossing, stale observation,
    insufficient history, excessive residual, insufficient deadline slack, and
    orientation mismatch all produce `HOLD`.
  - Commit is deterministic and target position remains immutable afterward.
  - A contradictory fresh observation aborts before movement or enters the
    defined safe hold behavior after commit.
  - Old committed evidence expires; it cannot refresh the arm deadman.
  - All transition APIs accept explicit `now_s` or an injected monotonic clock
    so boundary tests are deterministic.
- **Validation:**
  - Table-driven unit tests for every transition and rejection reason.
  - Synthetic tests for constant velocity, jitter, stop, reversal, occlusion,
    missed deadline, and already-passed crossing.

### Task 2.3: Expose estimator evidence

- **Locations:**
  - `src/object_tracking/arm_tracking/tracking.py`
  - `tests/test_depth_geometry_tracking.py` or
    `tests/test_interception_runtime.py`
- **Description:** Expose consecutive sample count and last innovation/residual
  norm from `PositionVelocityFilter` without changing its existing prediction
  behavior.
- **Perceived complexity:** 2/10
- **Dependencies:** None
- **Acceptance criteria:**
  - Reset clears sample count and residual.
  - Updates expose deterministic evidence for confidence gates.
  - Existing filter tests and behavior continue to pass.

## Sprint 3: Wire interception into the physical runtime

**Goal:** Dry-run produces a fully checked intercept candidate. Execute mode
publishes only after all existing and new gates pass.

**Demo/validation:** Run the existing fake transport/IK tests. In dry-run,
observe complete intercept telemetry with zero published targets. In execute
tests, only a fresh committed plan may publish.

### Task 3.1: Add disabled-by-default CLI and launcher plumbing

- **Locations:**
  - `src/object_tracking/yolo_stream_server.py`
  - `src/object_tracking/g1_cli.py`
  - `scripts/gb10/start.sh`
- **Description:**
  - Add `--intercept-config PATH`.
  - Add `INTERCEPT_CONFIG` launcher environment support.
  - No option means the existing tracking behavior.
  - A supplied option enables intercept preview; it does not imply execute.
- **Perceived complexity:** 3/10
- **Dependencies:** Task 2.1
- **Acceptance criteria:**
  - Old commands and tests behave identically without the option.
  - Missing or invalid config fails before creating an arm session.
  - Help text states that movement still requires both robot permission and
    explicit GB10 execute.
- **Validation:**
  - Extend `tests/test_g1_cli.py`.
  - Launcher shell syntax check.

### Task 3.2: Form a planner request from fresh measured state

- **Location:** `src/object_tracking/arm_tracking/runtime.py`
- **Description:**
  - Use the filtered object position/velocity and the exact observation time.
  - Obtain fresh measured right-arm joints from arm state.
  - Require the arm to be within the configured tolerance of the physically
    validated ready pose before the first commit.
  - Compute measured palm position and preserve the FK rotation for
    translation-only IK.
  - Reject stale measured state.
  - Call the pure intercept controller.
- **Perceived complexity:** 6/10
- **Dependencies:** Sprints 1 and 2
- **Acceptance criteria:**
  - Add a strict interception-specific measured-joint helper. The existing
    `right_arm_ik_seed()` commanded-first behavior remains for continuous
    tracking but is not used for interception.
  - Measured visualization state must be available, finite, exactly 29 DOF, and
    no older than 250 ms.
  - Commanded joints are never substituted for stale/missing measured joints.
  - Do not assume any FK transform column is the physical palm normal; that
    tool-axis convention is not documented in the repository.
  - Compare the configured, physically validated lane-facing normal with the
    incoming object direction as planning evidence. Require tomorrow's runbook
    to validate actual palm facing at the ready pose.
  - Translation-only operation never attempts real-time reorientation.
  - Planner defaults are overridden by the validated live profile.
- **Validation:**
  - Fake-state tests for fresh, stale, missing, and malformed measured state.

### Task 3.3: Apply geometry, IK, path, and command gates

- **Locations:**
  - `src/object_tracking/arm_tracking/runtime.py`
  - Existing `src/object_tracking/arm_tracking/ik_solver.py`
- **Description:**
  - Check calibrated workspace and support-plane clearance.
  - Preserve the measured/ready-pose wrist orientation.
  - Latch the Cartesian intercept point, never a joint target.
  - On each committed tick, use fresh measured joints and existing local
    translation IK to validate and publish only the next bounded edge.
  - Preserve the already commissioned ready-pose orientation. Reject an object
    direction inconsistent with the configured lane-facing normal.
  - In preview, expose the candidate only.
  - In committed execute state, publish bounded joint targets through the
    existing transport.
- **Perceived complexity:** 6/10
- **Dependencies:** Task 3.2
- **Acceptance criteria:**
  - No incremental edge bypasses workspace, support, joint, collision, TTL, or bridge
    checks.
  - An IK/edge failure produces `HOLD`, never a partial unsafe target.
  - Telemetry says `next_edge_collision_checked`; it must not claim the entire
    future path has been proven.
  - Continuous tracking mode remains unchanged.
- **Validation:**
  - Extend `tests/test_arm_tracking_runtime.py` for preview, committed publish,
    workspace rejection, orientation rejection, IK failure, and collision
    failure.

### Task 3.4: Implement bounded commit/revalidation/hold semantics

- **Locations:**
  - `src/object_tracking/arm_tracking/runtime.py`
  - `src/object_tracking/arm_tracking/interception.py`
- **Description:**
  - Before commit, update preview on every fresh frame.
  - At commit, latch target position.
  - After commit, fresh consistent observations may revalidate timing but may
    not move the target.
  - Brief palm occlusion may continue toward/hold the latched target only while
    the last genuinely confirming observation remains inside the 250 ms TTL and
    the bounded crossing window remains valid.
  - After crossing/expiry, hold then stop through the existing arm bridge.
- **Perceived complexity:** 7/10
- **Dependencies:** Task 3.3
- **Acceptance criteria:**
  - No post-crossing reversal.
  - No stale timestamp substitution.
  - Target loss before commit publishes nothing.
  - Target loss after commit cannot sustain motion beyond the existing TTL.
  - Stop/runtime shutdown remains a bounded release.
  - Terminal expiry calls the existing safe stop and requires manual
    re-enable/reset; there is no automatic return motion.
- **Validation:**
  - Fake-clock tests covering every boundary around commit, crossing, hold, TTL,
    deadman, and stop.

## Sprint 4: Make tomorrow operation-only

**Goal:** An operator can configure, dry-run, inspect, and—after physical
commissioning—execute without editing code.

### Task 4.1: Expose essential intercept evidence

- **Locations:**
  - `src/object_tracking/arm_tracking/runtime.py`
  - `scripts/gb10/web/unitree_dual_viewer.html`
  - `src/object_tracking/research_session.py`
  - `src/object_tracking/tuning_cli.py`
- **Description:** Expose:
  - state and rejection reason;
  - detector frame ID/age and inference latency;
  - measured position/velocity and residual;
  - crossing position/time;
  - arm flight time and conservative deadline slack;
  - target position and commit status;
  - next-edge IK/collision status;
  - command/acceptance age and following error.
- **Perceived complexity:** 4/10
- **Dependencies:** Sprints 1-3
- **Acceptance criteria:**
  - Operator can distinguish perception, timing, prediction, reachability, IK,
    path, arm-state, and command failures.
  - The UI never labels a single detector score as system confidence.
  - Research export contains enough evidence to score repeated trials.
  - HTML additions and percentile dashboards are P1; complete health JSON and
    research telemetry are the P0 operator surface.
- **Validation:**
  - API/status unit tests.
  - Browser payload smoke test; UI polish is not a movement dependency.

### Task 4.2: Add the morning runbook

- **Locations:**
  - New `docs/REALTIME_INTERCEPT_MORNING_RUNBOOK.md`
  - `configs/intercept/demo-lane.example.yaml`
- **Description:** Provide copy/paste commands for:
  1. artifact and commit verification;
  2. robot observations/disarmed startup;
  3. GB10 alpha-beta dry-run;
  4. lane-profile validation;
  5. passive crossing capture;
  6. static ready-pose and intercept-pose gates;
  7. slow execute only after all gates pass;
  8. stop and log export.
- **Perceived complexity:** 2/10
- **Dependencies:** Completed CLI
- **Acceptance criteria:**
  - Dry-run is the default.
  - Automatic `bunny-test` is explicitly deferred until static commissioning
    and loss/deadman tests pass.
  - Every execute command includes the required robot and GB10 permission
    gates.

## Automated testing strategy

### Targeted tests

Run after each sprint:

```bash
.venv/bin/pytest -q \
  tests/test_intercept_planner.py \
  tests/test_interception_runtime.py \
  tests/test_arm_tracking_runtime.py \
  tests/test_depth_geometry_tracking.py \
  tests/test_ros2_tracking.py \
  tests/test_arm_bridge.py \
  tests/test_g1_cli.py
```

### Full verification

1. Install/use the complete development, vision, and arm environment needed to
   collect every test.
2. Run the complete non-hardware suite.
3. Run Ruff only on touched files and new tests; record unrelated pre-existing
   repository lint separately.
4. Run deterministic synthetic crossing tests over:
   - speeds from 0 to 0.20 m/s;
   - multiple observation latencies up to and beyond 250 ms;
   - jitter, dropped frames, duplicate/out-of-order frames, track switches,
     stop/reversal, occlusion, unreachable crossings, and expired crossings.
5. Repeat timing/state-machine tests enough times to detect flaky clock
   boundaries.
6. Inspect the complete diff for accidental safety-gate weakening.

### Verification completion criteria

- All pre-existing non-hardware tests pass with required optional dependencies.
- All new tests pass.
- Continuous tracking behavior is unchanged without `--intercept-config`.
- Dry-run never publishes a target.
- Execute tests never publish without `COMMITTED`, fresh
  measured state, valid calibration, next-edge IK/collision success, and existing bridge
  authorization.
- No test or implementation fabricates timestamp freshness.
- No configured failure can sustain commands beyond TTL/deadman bounds.

## Debugging reserve

Use the two-hour reserve in this order:

1. Timestamp-domain and frame-order failures.
2. State-machine boundary and stale-hold failures.
3. Runtime fake-transport/IK integration failures.
4. CLI/config compatibility failures.
5. Full-suite regressions.
6. UI/status issues.

Do not spend the reserve on performance tuning without live GB10 measurements,
new predictors, dependency upgrades, or visual redesign.

## Risks and mitigations

- **Two-hour coding scope is aggressive.** The reviewed estimate for the full
  plan is five to eight coding hours. Enforce the P0 cut line, keep UI and
  percentile aggregation optional, and allow the debugging reserve to absorb
  correctness-related implementation overrun.
- **Planner flight-time defaults are not physical evidence.** Require a
  profile and show the assumed values; tomorrow replaces them with measured
  conservative values.
- **Palm orientation may not face the lane.** Require a static validated ready
  pose and reject runtime orientation changes.
- **Palm occlusion can hide the bunny near contact.** Latch the target, allow
  only bounded evidence revalidation/hold, and retain the 250 ms freshness
  ceiling.
- **A fixed lane can still have lateral error.** Configure a permitted crossing
  segment and tomorrow calculate the spatial error budget from palm width,
  lane error, calibration error, arm error, and prediction error.
- **Robot mechanics are unavailable tonight.** Do not claim physical latency,
  endpoint accuracy, or safe contact until tomorrow's staged tests.

## Rollback plan

- The new mode is disabled unless `--intercept-config` is supplied.
- Removing that option restores the current continuous tracking runtime.
- Keep changes in atomic commits:
  1. frame provenance and timing;
  2. pure intercept controller and tests;
  3. runtime/CLI integration;
  4. telemetry/runbook.
- Do not modify or delete calibration, model, home, or recorded-run artifacts.
- If execute integration is not fully verified, freeze the dry-run preview and
  retain the existing commissioned static-motion path for hardware testing.

## Explicit non-goals tonight

- Physical robot authorization or unsupervised movement.
- MoveIt migration.
- MPC, proportional-navigation, VLA, or a new learned model.
- Dynamic wrist orientation.
- Arbitrary target trajectories.
- Grasping.
- Relaxing TTL, deadman, collision, workspace, or following-error gates.
- Tuning gains without measured hardware telemetry.
