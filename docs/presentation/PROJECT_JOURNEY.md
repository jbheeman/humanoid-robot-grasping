---
title: Explain how the project evolved
contentType: Conceptual
---

# Explain how the project evolved

The project evolved from a camera and locomotion prototype into a guarded, distributed robot-learning system. Each stage exposed the next bottleneck: perception rate, safe actuation, geometric planning, temporal prediction, and finally end-to-end latency.

This page is the presentation-length narrative. [The implementation and branch history](IMPLEMENTATION_HISTORY.md) traces the actual commit sequence across `aarav`, `aarav-vla`, `sim`, `vla-sim`, `lightweight-intercept-isaac`, and the smaller contributor branches.

## Branches shaped different parts of the system

The physical stack grew mainly on `aarav` until July 20. `aarav-vla` then continued from the same base with guarded VLA execution, data curation, training automation, analytic IK, real-time interception, and Ruckig trajectory shaping. In parallel, `sim` built controller tuning, `vla-sim` extended it with Isaac dataset generation, and `lightweight-intercept-isaac` added privileged oracle evaluation.

The smaller `neel`, `neel2`, `calibration`, and `oliver` branches captured early locomotion, SDK, calibration, and hand-control experiments. `main` and `advay` remained at the initial repository stage.

The current `demo-realtime-intercept-minimal` branch starts from the final `aarav-vla` implementation. Its first commit removed 25,756 lines of training, legacy control, and overlapping tooling. It retained the shortest path needed for a bounded physical interception demo.

## Project goal

The practical goal is to move the Unitree G1’s right arm toward a moving plush bunny using live vision while keeping every command inside robot, joint, collision, and tabletop safety limits. In the current hardware configuration, “success” means controlled palm contact or interception, not a verified fingered grasp.

## Phase 1: Establish vision and robot connectivity

**Dates:** July 6–8, 2026

The first phase established the basic development loop:

- Created the Python project and object-tracking scaffold
- Added prototype G1 movement and locomotion commands
- Replaced a heavy live stream with a UDP GStreamer camera relay
- Added Streamlit and YOLO-based visualization
- Built DDS discovery and robot-local diagnostic tools

The main achievement was a repeatable path from the robot camera to a workstation that could run object detection and display results.

## Phase 2: Split perception from robot control

**Dates:** July 9–14, 2026

The team moved compute-intensive perception to the GB10 and kept hardware-facing control on the robot:

- Bound the robot’s RealSense and main camera streams reliably
- Targeted 30 FPS, then 60 FPS, perception
- Added YOLO training launchers and TensorRT export
- Added guarded arm commissioning, telemetry capture, and joint audits
- Migrated project control from HTTP bridges to ROS 2
- Added depth transport, tabletop localization, and browser-based calibration
- Trained a synthetic gated recurrent unit (GRU) trajectory forecaster

This stage established the safety principle that remains in the current design: the robot owns the final command guard, while the workstation proposes high-level arm targets.

## Phase 3: Make the physical arm follow the bunny

**Dates:** July 15–16, 2026

The team progressed from joint-level movement to geometric tracking:

- Verified remote dual-arm control and conservative manual moves
- Added Cartesian IK and local waypoint execution
- Added full-link collision reporting and collision-aware approach planning
- Combined RGB detection, depth, tabletop geometry, and measured arm state
- Added continuous, latest-target-wins vision tracking
- Logged physical tracking sessions for later analysis

**Physical result:** The robot moved well and tracked the bunny with IK. The limiting issue was speed, not basic feasibility. Planning and command updates were too slow for a robust moving-target interception.

This result is team-confirmed hardware behavior. The repository contains many tracking telemetry sessions, but it does not contain a single standardized “success rate” for those physical trials.

## Phase 4: Add learned action proposals

**Dates:** July 20–21, 2026

The project integrated Unitree’s UniFoLM-VLA model:

- Added inference-only and guarded-execution modes
- Kept the model resident to reduce startup overhead
- Restricted VLA output to bounded right-arm motion
- Routed learned proposals through the same geometric IK and collision gateway
- Added a persistent read-only observation bridge

The VLA never replaced the safety controller. It proposed task motion, while deterministic geometry and robot-side deadman checks remained responsible for safe execution.

## Phase 5: Build the sim-to-real data pipeline

**Dates:** July 21–23, 2026

The dual-3090 machine became the main dataset and training host. The team:

- Built an Isaac Sim moving-bunny scene and capture pipeline
- Iterated through more than 20 synthetic scene and contact variants
- Generated 480 moving-block simulation episodes in dataset v28
- Audited 127 raw real demonstration episodes
- Materialized 97 accepted or reviewed real episodes without modifying raw data
- Applied stricter motion, timing, and contact checks to produce 67 canonical real episodes
- Created leakage-safe train, validation, and sealed test splits by capture session
- Converted the data to the RLDS and TensorFlow Datasets formats expected by UniFoLM

The latest combined canonical dataset contains 547 episodes: 480 Isaac Sim episodes and 67 real XR-teleoperation episodes.

## Phase 6: Diagnose why the VLA did not generalize

**Date:** July 23, 2026

The first complete VLA schedules produced checkpoints, but the learned trajectories did not beat “hold the current pose” or mean-action baselines. The team then ran controlled ablations:

- Absolute actions versus anchored relative actions
- Single-frame observations versus five-frame history
- Full 23-dimensional actions versus right-arm-only actions
- One-frame and three-frame future targets
- Translation-only objectives and rotation down-weighting
- Real/synthetic sampling ratios and action-head warm starts

The key discovery was a target-semantics problem. Training against recorded commands did not match the state the robot achieved after observation and command latency. Rebuilding targets around the achieved future state improved the weighted right-hand average displacement error from 0.08535 m to 0.06695 m, a 21.6% improvement.

The corrected model still failed the baseline gate. Its predicted displacement remained too large, and about 8.8% of normalized outputs saturated. This finding redirected the work from “train longer” toward timing, action calibration, and closed-loop evaluation.

## Phase 7: Automate training and reject weak checkpoints

**Dates:** July 23–24, 2026

The team built restart-safe training, evaluation, and promotion automation:

- Evaluated every checkpoint on validation data
- Kept the test split sealed until model selection
- Added GPU failover after GPU 0 instability
- Added an adaptive 24-attempt campaign with patience and retention rules
- Recorded metrics in a queryable campaign ledger
- Prevented validation failures from reaching robot execution

The v29 adaptive campaign completed 22 attempts and paused for no progress. Its latest weighted error was 0.06772 m, close to the best prior result, but every candidate failed baseline, magnitude, or saturation gates.

The v30 active-motion experiment increased the proportion of informative moving windows. Training was interrupted by distributed GPU failures and a later evaluation configuration error. It produced no promoted checkpoint.

## Phase 8: Remove the real-time IK bottleneck

**Dates:** July 24–27, 2026

The team replaced the slower finite-difference IK update with an analytic Pinocchio Jacobian servo:

- Median local IK step time fell from 0.136 ms to 0.080 ms in the offline benchmark
- The analytic version reduced the Jacobian step latency by about 41%
- All 64 benchmark targets produced accepted bounded steps
- The real-time interception runtime now tracks frame provenance, RGB-depth timing, target freshness, and commit/revalidation state

The July 27 dry-run reached about 57 camera FPS and 49–51 YOLO FPS in representative sessions. Safety gates correctly rejected motion when the support plane was unavailable or the initial arm state intersected the hip collision margin.

## Current phase: Final physical retest

The last demonstrated physical state is successful but slow IK bunny tracking. The faster analytic servo, Ruckig trajectory shaping, and revised interception runtime have been implemented but have not yet completed the equivalent robot retest.

The next test should answer one question: does the new fast path preserve the previous tracking quality while reducing enough latency to intercept a moving bunny safely?

Success requires:

1. Stable 60 FPS-class RGB capture and fresh paired depth
2. A valid measured arm state and collision-free escape posture
3. Continuous tracking without stale-frame commands
4. Faster bounded arm response than the earlier physical controller
5. No safety-gate bypass, table strike, self-collision, or uncontrolled command
