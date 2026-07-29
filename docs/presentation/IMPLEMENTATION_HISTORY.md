---
title: Trace the implementation across every branch
contentType: Conceptual
---

# Trace the implementation across every branch

The repository history is not one linear build. The physical robot stack, controller simulation, Isaac dataset generation, and VLA experiments developed on parallel branches that forked and occasionally rejoined through copied artifacts or later implementations. This page reconstructs that implementation history from every GitHub branch and its commit graph.

## How the branches relate

The branch graph has three main lines:

```text
main / advay
│
├── calibration                 early standalone arm-calibration controls
├── neel / neel2                early locomotion and robot-control experiments
├── oliver                      separate Dex3-1 finger-joint experiment
│
└── aarav lineage               main physical perception and robot-control work
    │
    ├── sim                     MuJoCo/MJX controller search and validation
    │   └── vla-sim             Isaac bunny data and RLDS/TFDS pipeline
    │       └── lightweight-intercept-isaac
    │                           oracle interception evaluation
    │
    ├── aarav                   physical stack plus three GB10 experiments
    │
    └── aarav-vla               guarded VLA, data curation, training,
        │                       analytic IK, and real-time interception
        └── demo-realtime-intercept-minimal
                                current focused hardware-demo path
```

`aarav` and `aarav-vla` share history through commit `7276cd9` on July 20. `sim` forked much earlier at `e2c897a` on July 10. `vla-sim` extends `sim`, and `lightweight-intercept-isaac` extends `vla-sim`. The current `demo-realtime-intercept-minimal` branch extends the July 27 `aarav-vla` tip by two commits.

## What each branch contributed

The counts below describe the remote-tracking branches after refreshing all refs on July 27, 2026.

| Branch | Commits at tip | Relationship and contribution |
| --- | ---: | --- |
| `main` | 2 | Repository creation and README only |
| `advay` | 2 | Same initial history as `main`; no later implementation |
| `neel2` | 33 | Early movement, SDK example, and interactive-drive work that entered the shared `aarav` history |
| `neel` | 37 | Continued the early control line with two branch-only dependency/drive cleanup commits |
| `calibration` | 37 | Early standalone G1 calibration controls and dependency setup |
| `oliver` | 11 | Independent early work plus a Dex3-1 single-finger joint helper |
| `aarav` | 219 | Primary physical perception, commissioning, ROS 2, IK, and tracking history through July 20 |
| `sim` | 116 | 34 branch-specific commits for parallel MuJoCo/MJX controller optimization and robust validation |
| `vla-sim` | 131 | All `sim` work plus 15 commits for Isaac capture, bunny assets, and UniFoLM dataset conversion |
| `isaac/vla-sim` | 131 | Cached remote-tracking ref identical to `origin/vla-sim`; no additional commits |
| `lightweight-intercept-isaac` | 134 | All `vla-sim` work plus three oracle-interception evaluation commits |
| `aarav-vla` | 308 | Shared physical stack plus 92 commits for guarded VLA, real-data curation, training, faster IK, live interception, and jerk-limited execution |
| `demo-realtime-intercept-minimal` | 310 | Current path: a focused reduction and high-FPS live-test update on top of `aarav-vla` |

The large branches changed different surfaces:

- `aarav`: 188 files and about 36,800 inserted lines after the initial README
- `sim`: 27 files and about 26,200 inserted lines after its July 10 fork
- `vla-sim`: 45 additional files and about 4,800 inserted lines beyond `sim`
- `lightweight-intercept-isaac`: eight files and about 900 inserted lines beyond `vla-sim`
- `aarav-vla`: 92 commits after its July 20 fork from `aarav`
- `demo-realtime-intercept-minimal`: removed about 25,756 lines across 183 files while adding 92 focused lines

## July 6: Create the repository

The `main` and `advay` branches stop after the initial repository and README commits (`8a85b1f`, `f6b3e7f`). They establish the project location but contain no robot implementation.

This matters when presenting the history: almost all engineering happened on feature branches rather than `main`.

## July 7: Build the first end-to-end prototype

The early `aarav`/`neel2` line moved rapidly from a scaffold to a connected demo:

1. `7641abc` added the object-tracking scaffold
2. `7a479b0` added prototype movement controls
3. `4e161f5` wrapped vision and locomotion as `uv` applications
4. `79fe045` added SSH support and moved video to UDP GStreamer
5. `c9a01ef` added Streamlit visualization

The architecture was still exploratory. Vision, remote access, movement, and display were being proven independently rather than treated as a single safe control loop.

## July 8: Debug Unitree connectivity and create the first control boundary

Thirty-seven commits on the shared line show how much integration work was required before higher-level behavior:

- `855e6a0` through `8168d08` optimized YOLO capture and added RealSense support
- `43b2e1c` through `d1653af` added DDS probes, motion-switcher commands, RPC decoding, and robot-local diagnostics
- `933297d` added a DDS discovery wrapper
- `2a7a145` created an HTTP motion bridge after direct DDS experimentation
- `f7e8e0d` and `05b371f` added bounded arm-movement tests
- `b93ecc8` consolidated robot commands and safety flags
- `0f5a409` added interactive drive and software stop

Commit `12441b7` merged the `neel` work into `aarav`. Later, `60c2956` removed interactive drive from the main line, showing an early design correction: unrestricted operator control did not fit the emerging safety model.

The separate branches explored alternatives:

- `calibration` added standalone arm calibration (`0b8277e`) but did not become the long-term commissioning implementation
- `neel` retained branch-only drive cleanup and dependency changes (`7afb446`, `78af511`)
- `oliver` later added a Dex3-1 finger-joint helper (`c17e19f`), which remained separate from the no-hand bunny-tracking runtime

## July 9: Separate robot capture from workstation inference

The next 25 commits formed the first recognizable distributed system:

- Split dependency groups for robot, vision, training, and locomotion
- Added robust camera-device mapping and binding diagnostics
- Split robot image capture from GB10 inference in `42ed7bb`
- Restored and validated the Unitree multicast relay
- Raised plushie tracking to 30 FPS

The day then shifted from perception to manipulation:

- `1171dfe` documented RealSense-guided arm tracking
- `b1e8ac0` implemented it
- `f67c514` added research telemetry and run capture
- `1e3cdd5` added joint-contract audits and dry-run tuning
- `b7b33f4` added guarded right-arm commissioning
- `91146d9` introduced the unified `g1` operator command

This was the first point where the repository contained perception, robot actuation, safety gates, and evidence capture as parts of one workflow.

## July 10: Harden commissioning and start detector training

The July 10 commits focused on making the system usable in the lab:

- Added arm visualization and improved tracking logic
- Fixed robot-side Python 3.8 compatibility
- Added role-specific convenience launchers
- Added wired commissioning
- Prevented heartbeats after failed enable
- Limited initial commissioning to shoulder joints
- Preserved the original commissioning fault for diagnosis

The same day added YOLO11s 60 FPS and YOLO26l accuracy training launchers (`cacef86`, `e2c897a`). Commit `e2c897a` became the fork point for the later `sim` family of branches.

## July 11–14 on `sim`: Search for robust arm-control parameters

While the physical branch continued separately, `sim` added 34 branch-specific commits.

The first simulation implementation used a suspended G1 arm (`2733c04`). It then evolved through:

- Parallel MuJoCo workers and observable progress
- MJX GPU batching with memory caps
- Adaptive batch sizes under shared VRAM pressure
- Elite controller ranking and exchange across hosts
- CPU fallback and memory-pressure pauses
- Randomized delay, torque, position-noise, and velocity-noise validation
- Commissioning telemetry fitting
- Plateau-based stopping and robust replay scoring

The sequence from `15f3fa1` through `8cf2ee5` is especially revealing. Much of the work was not controller math; it was preventing GPU memory pressure, process leaks, and shared-host contention from invalidating long searches.

Commits `a4529d3` and `fd5bbda` added robust candidate validation. `dee4f4e` recorded the verified six-hour run later summarized as 9,312 evaluated trials.

The last five `sim` commits brought browser localization improvements into the simulation line. They remained separate hashes from similar physical-branch changes.

## July 11–14 on `aarav`: Replace provisional control with ROS 2

The physical line first added offline bunny future-position evaluation (`edd46cd`). On July 13, it began a 24-commit ROS 2 migration:

- `9946ca9` migrated robot control to ROS 2
- `2da1a10` added Foxy CycloneDDS peer configuration
- `2301b3f` moved final robot control to native Unitree DDS
- `4aaa460` separated project ROS middleware from native Unitree middleware
- `1eb51aa` and `c2bb656` isolated their domains and startup defaults
- `76de53f` added the persistent high-FPS camera service
- `bd9c966` and `df31bf5` bound the bridge to the correct wireless route

The second half of the day added the first direct plushie tracker:

- `ebb20e1` added bounded visual-lock tracking
- `9d6e163` moved it to a fixed-rate guarded control loop
- `04b5915` used the D435I factory camera geometry

This migration replaced the early HTTP control experiment with the ROS 2 and robot-local safety boundary used by later work.

## July 14: Turn depth into usable tabletop geometry

Thirty-eight physical-branch commits developed the RGB-depth-localization chain:

- Probed supported RealSense depth profiles
- Added raw depth preview and trajectory-forecast evaluation
- Added D435I localization validation and interactive tuning
- Recovered stalled depth capture
- Bounded depth publication to 15 Hz
- Added aligned RGB tracking and tabletop corner calibration
- Localized plush tracks inside the tabletop map
- Added a browser localization tuner

The roadmap changed several times in commits `967f8ac` through `c1f1750`: from broad interception, to bounded image-plane pointing, to floor-plane calibration, and finally to tabletop homography. Those changes document a deliberate reduction in scope toward something calibratable on the real robot.

## July 15: Progress from joint motion to continuous collision-aware tracking

Sixty-five commits capture the most intensive physical implementation day.

### Establish reliable remote arm control

The first sequence:

- Added a standalone arm movement test
- Added ROS 2 remote dual-arm control
- Isolated manual arm endpoints
- Standardized the cross-version Fast DDS channel
- Latched measured baselines and hardened actuation
- Added visible, inverted, and compact multi-joint moves

### Add Cartesian IK

The next sequence progressed in small verified steps:

- `971a7f2` added a guarded Cartesian IK test
- `22b6706` built the IK model before arming
- `2db1c8e` tightened accuracy
- `3b82454` relaxed wrist orientation where necessary
- `2477f8c` expanded trials to 5 cm
- `a278b6b` solved through local waypoints
- `a15041e` executed the waypoint path
- `29d1a43` reported colliding link pairs

This progression explains why the first physical tracker was slow: each target could require global solving, waypoint construction, collision evaluation, and guarded streaming.

### Connect IK to vision

Commit `db75ac2` added guarded vision pointing. The following commits restored the native high-FPS RGB path, isolated the vision UI from arm transport, separated depth and arm DDS domains, and persisted debugging logs.

The planner then became collision-aware:

- `7fedbb2` planned collision-aware routes
- `b38493c` preplanned the full approach on the GB10
- `ec32e86` hardened those routes
- `fb7f437` executed and replanned stages incrementally
- `3b3c828` routed the arm through a hip-clearance posture

### Convert pointing into tracking

The final sequence replaced one-shot pointing with continuous behavior:

- Kept targets alive across replanning
- Anchored commands to the last accepted arm target
- Increased arm waypoint and depth throughput
- Removed the depth filter bottleneck
- Smoothed trajectories
- Bounded route-planning latency

This is the implementation history behind the team-confirmed result that the robot moved well and followed the bunny, but reacted too slowly.

## July 16: Make tracking responsive and table-aware

Five commits consolidated the physical tracker:

- `af41616` added latest-target-wins servoing
- `45e657e` required a live support plane
- `f2303a0` added bounded tabletop clearance
- `9c42332` allowed safe near-edge pointing
- `7842932` added Ubuntu 22/Humble workstation support

`7276cd9` on July 20 then hardened vision pointing and ROS transport. This commit is the last shared ancestor of `aarav` and `aarav-vla`.

## July 15–22 on `vla-sim`: Build the Isaac and data-conversion line

`vla-sim` extends all of `sim` with 15 commits:

- `b6991e3` added the G1 bunny UniFoLM dataset pipeline
- `8a00128` added headless Isaac smoke capture
- `914381f` and `354f11a` fixed Isaac startup and imports
- `00a707c` and `47f8582` made RLDS and TensorFlow Datasets conversion reproducible
- `42ee37e` through `cd318b8` built and visually corrected a measured bunny proxy
- `5deb024` through `0dad7bc` added restart-safe overnight asset validation across both GPUs
- `4705cd1` built the realistic moving-rabbit dataset pipeline

This branch explains where the synthetic dataset work originated. The later `aarav-vla` branch implemented its own orchestration and training-facing copies, but the Isaac scene and asset work began here.

## July 20: Split `aarav` from `aarav-vla`

After `7276cd9`, the two branches pursued different work.

`aarav` added three branch-only experiments:

- `f8b7e60`: versioned GB10 UniFoLM training adaptations
- `2e76cc9`: added a local GB10 voice bridge
- `cb7deb3`: displayed registered torso position in the GB10 viewer

`aarav-vla` instead began the guarded learned-policy path:

- `0c9958a` added guarded UniFoLM-VLA execution
- `add1bcd` preloaded the model
- `90ce31c` clarified and guarded task proposals
- `78472cf` handled quantized checkpoints
- `f16bcfe` added a trained-checkpoint launcher

The split was therefore not “old branch versus new branch.” It was a product choice: `aarav` retained GB10 interaction experiments, while `aarav-vla` concentrated on learned action proposals and eventual interception.

## July 21–22 on `aarav-vla`: Separate processes and add moving-target semantics

The VLA branch then:

- Added a persistent observation bridge
- Split the native Unitree arm worker from the ROS relay
- Fixed guarded clearance execution
- Added bounded direct-VLA smoke modes
- Kept the model resident after interruption
- Added a latency-aware moving-rabbit VLA pipeline
- Hardened moving-block dataset orchestration

The native-worker split (`f0129fd`) was an architectural turning point. Unitree DDS and project ROS no longer needed to coexist in one Python interpreter.

## July 23 on `aarav-vla`: Turn failed training into an experiment program

Thirty-five commits converted ad hoc model training into a controlled research pipeline.

The sequence was:

1. Separate learned task behavior from deterministic geometry safety
2. Add event-driven pilot training and evaluation
3. Compare absolute and anchored relative Pose23 actions
4. Recover from GPU and evaluator failures
5. Diagnose output magnitude and saturation
6. Align labels to achieved future state
7. Verify exact TFDS target semantics
8. Compare translation-only, rotation-weighted, and absolute-future targets
9. Add latency-safe evaluation
10. Curate and visually review real episodes
11. Apply leakage-safe, session-held-out splits
12. Automate validation, rejection, retries, and adaptive campaigns

This history is why the 21.6% timing-alignment improvement is credible. It followed several failed representations and explicit data-contract checks rather than appearing as one successful training run.

## July 24 on `aarav-vla`: Optimize the controller and assemble the live test

Thirty-eight commits pursued two tracks in parallel.

### Training infrastructure

- Added active-motion v30 data and configuration
- Supervised dual-GPU health
- Added failover and campaign handoff

### Real-time geometric control

- `f063dbe` integrated the analytic Pinocchio IK servo
- `d6c743b` added latency-aware interception evaluation
- `3ee9d76` required complete safety instrumentation
- Added surface standoff and measured-pose warm starts
- Added a streamed hip-rest escape corridor
- Latched the escape across collision-sensor flicker
- Bridged short support-plane dropouts
- Froze a validated table plane during armed motion
- Added one-command bunny testing with a fresh disarmed bridge
- Kept targets alive while the robot armed
- Stabilized synchronized 60 Hz RGB and depth ownership
- Changed the depth wire format to survive Foxy/Jazzy transport

The many support-plane, hip-escape, and depth commits show that the final bottleneck was system coordination, not a single IK formula.

## July 24 on `lightweight-intercept-isaac`: Add an oracle baseline

This branch extends `vla-sim` with three commits:

- `2fb2a49` added latency-aware bunny-intercept evaluation
- `1d35bdf` added unbiased Isaac oracle rollouts
- `ff5e448` stratified and packaged the oracle pilot

Its role is to answer whether the simulated task and evaluator are solvable with privileged target information before blaming the learned policy. It is an evaluation branch, not the deployed robot runtime.

## July 27: Assemble and smooth the real-time path

The final `aarav-vla` sequence contains six changes:

- `f70ae89` added the real-time moving-object interception runtime
- `2317771` matched the GB10 tracking middleware with the robot bridge
- `c67ca69` selected the calibrated GB10 camera profile by default
- `c982321` added jerk-limited autonomous arm trajectories with Ruckig
- `90f56f0` pinned the Ruckig build tools required by the G1 Python environment
- `c5c3366` exposed the active Ruckig limits in bridge health

These commits complete the control path intended for the final robot retest. Ruckig shapes each accepted joint target into bounded velocity, acceleration, and jerk updates. The commits do not prove the retest result by themselves.

## July 27: Create the focused current branch

Commit `d0a873f` created `demo-realtime-intercept-minimal` directly from `c5c3366`. This branch is the current working path.

The commit removed broad research surfaces that are not required for the final demo:

- UniFoLM training, data curation, and evaluation scripts
- RLDS and TensorFlow Datasets conversion code
- Legacy manual-control and calibration entry points
- Older viewers, launchers, and experimental robot trackers
- Tests that covered removed research-only modules

The branch retained the focused runtime:

- RealSense RGB and depth capture
- YOLO bunny detection
- 3D localization and velocity tracking
- Plane-crossing interception logic
- Analytic Pinocchio inverse kinematics
- Collision and tabletop validation
- Ruckig trajectory shaping
- Robot-side deadman, state, and command guards
- The GB10 dashboard and research telemetry

The reduction changed 183 files, added 92 lines, and removed 25,756 lines. It did not replace the research history. It converted the proven components into a smaller demo surface with fewer competing launch paths and dependencies.

Commit `97b9853` then made the live bunny test use the root-owned 960×540 at 60 FPS camera service. It separated high-rate RGB encoding from the RealSense depth owner, required eight consecutive healthy readiness samples, and stopped treating a single healthy response as sufficient to arm the test.

## What the commit history says about the project

The implementation journey has four recurring patterns:

1. **Start with a bounded hardware primitive.** Joint jogs came before Cartesian paths; paths came before tracking; tracking came before interception.
2. **Move unstable work offline.** Controller search moved to MuJoCo/MJX, visual behavior to Isaac, and policy comparison to held-out evaluation.
3. **Treat failures as contract evidence.** DDS crashes, depth stalls, GPU loss, target misalignment, and output saturation each produced a specific guard or experiment.
4. **Keep learned behavior behind deterministic safety.** No branch allowed an offline metric to bypass collision geometry, stale-state checks, or robot-side command ownership.

The final project state follows directly from that history: physical tracking is demonstrated, the fast interception path is implemented, and its equivalent hardware validation remains pending.
