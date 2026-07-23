# Dynamic-rabbit VLA recovery

## Iteration compact

- **Goal:** Produce a right-hand policy that blocks the moving rabbit. A
  separate geometric IK gateway—not the learned objective—enforces table and
  robot safety.
- **Decision owner:** The robot operator. Model promotion never authorizes
  physical movement.
- **Primary metric:** Physics-qualified right-palm/rabbit block success in
  non-paused held-out simulation.
- **Guardrails:** Every proposed chunk passes through live finite-table
  geometry, half-space projection, full-link swept collision checking,
  joint/rate limits, and fail-closed braking. No execution path may publish
  around that gateway.
- **Unacceptable mistakes:** Moving from stale imagery, violating table
  clearance, treating operator trigger state as physics contact, or selecting a
  checkpoint using the test split.
- **Data snapshot:** `moving_block_480_v28` plus
  `plush_touch_canonical_block_v28`; 480 synthetic episodes and 15 strictly
  curated real episodes.
- **Baseline:** `block-v28-real-1k/steps_250_action_model.pt`; sealed real-test
  right-XYZ ADE 0.1443 m and FDE 0.1429 m. It is not deployable.
- **Main hypotheses:** The current single-frame policy cannot infer rabbit
  velocity; capture-to-action latency is not represented explicitly; absolute
  23-D targets and real oversampling amplify the sim-real mismatch.
- **First experiment:** Verify timestamp/action/frame contracts and compare
  constant, current-pose, and learned predictions before changing the model.
- **Rollback:** Retain the immutable v28 data and failed checkpoint. New
  derived datasets and runs receive unique IDs and cannot overwrite them.

## Immutable gates

1. Ground-truth action/state alignment has a unique causal lag and reconstructs
   the recorded trajectory without unit or frame violations.
2. Physics contact is the only success label for synthetic episodes.
3. Demonstrations teach interception behavior; table clearance is not a
   supervised label or model-selection metric. Grossly infeasible
   demonstrations are still excluded.
4. A learned model must beat constant/current-pose baselines on held-out data.
5. Closed-loop held-out simulation must reach 80% block success before any
   user-operated hardware experiment.
6. The runtime gateway must maintain its configured table/tool clearance with
   zero unsafe contacts, including direct VLA mode and fault injection.
7. The real test split is evaluated once after model selection.

## Experiment ladder

1. Freeze the action/time/geometry contract and prove that all command paths
   use the same safety gateway.
2. Audit the existing 1-frame, 25-action, absolute-EE model against
   current-pose, mean-action, frozen-image, and shuffled-image baselines.
3. Qualify live-plane projection plus continuous, full-link IK validation in
   mock and non-paused simulation, including stale-plane and invalid-target
   faults.
4. Compare 1-frame history with five timestamped frames spanning 0.3–0.5
   seconds; overlap inference and receding-horizon execution.
5. Compare full absolute EE, right-only absolute EE, and right-relative EE
   action targets at equal optimizer budgets.
6. Compare frozen visual features with a small visual adapter only after a
   rabbit-position/motion probe demonstrates that adaptation is necessary.
7. Treat official Unitree data as a low-weight regularization ablation only
   after its joint/frame/rate/normalization adapter round-trips exactly.
8. Expand synthetic data or introduce simulation RL only after the preceding
   gates pass.

## Runtime safety qualification

Before a physical trial, the gateway is qualified independently of model
accuracy:

- The tabletop is a bounded footprint with a conservative lateral margin, not
  an infinite plane. Its live height, normal, dimensions, freshness, and
  camera-to-torso transform must agree with calibration.
- The installed no-hand tool profile and future BrainCo hand profile each need
  their own collision geometry and tool transform. A profile mismatch is a
  startup failure.
- The analytic palm projection is followed by full-link swept collision
  validation; end-effector projection alone is never treated as proof of
  safety.
- Joint position, per-tick step, self-collision, table footprint, contact,
  stale state, stale geometry, and out-of-order chunk faults all fail closed.
- A stale signal during motion must invoke the robot bridge's bounded stop
  ramp, then hold. The VLA controller itself cannot refresh that deadman.
- Nominal ground-truth trajectories should require projection on less than 2%
  of commands. Frequent correction means the model/action contract is wrong,
  not that the safety layer is working well.

Physical ROS and Unitree SDK processes are intentionally outside this
implementation workflow.

## Why safety is outside the learned objective

Unitree's public G1 manipulation data is ordinary 30 Hz imitation data with
robot observations, actions, and camera streams; it does not expose a learned
"do not hit the table" label.  That matches the intended split here:

- UniFoLM learns which task motion to attempt.
- The live table estimator supplies a bounded tabletop in the torso frame.
- `GeometricIKGateway` minimally projects a palm target into the table's safe
  half-space, then requires full-link continuous collision and joint-limit
  validation.
- `SafeVLAControllerCore` revalidates one receding-horizon waypoint against the
  current table and arm state. Stale geometry, contact, malformed chunks, or IK
  failure cancel the active chunk.

The training data must still contain physically meaningful successful
interceptions. Runtime filtering is a safety boundary, not a way to repair a
systematically incorrect action convention.

## Current diagnostic finding

On 24 deterministic validation windows per source, the selected v28 checkpoint
remained worse than holding the current pose:

- Real: 0.1158 m learned ADE versus 0.0714 m current-pose ADE.
- Synthetic: 0.1360 m learned ADE versus 0.0337 m current-pose ADE.
- Image shuffling and occlusion changed predictions by roughly 0.10–0.11 m, so
  the model is visually conditioned.
- The main failure is action calibration: synthetic predictions moved about
  0.148 m from the current pose while targets moved only about 0.034 m, with
  roughly 10% normalized-output saturation.

The next model experiment therefore targets temporal motion information and
action magnitude/normalization. It does not add a table-clearance loss.

## Research basis

- [VLSA / AEGIS](https://arxiv.org/abs/2512.11891) treats geometric safety as
  a plug-and-play control-barrier layer around a VLA, rather than as another
  behavior label.
- [Path-Consistent Safety Filtering](https://arxiv.org/abs/2511.06385) argues
  that a safety filter should preserve the policy's intended path where
  possible. This is why the gateway applies a minimal plane-normal projection
  before rejecting an unsafe chunk.
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
  motivates asynchronous inference and receding-horizon chunk consumption for
  inference-limited VLAs.
- [SmolVLA](https://arxiv.org/abs/2506.01844) independently reports the value
  of asynchronous inference for responsive robot control.
- Unitree's [UniFoLM-VLA repository](https://github.com/unitreerobotics/unifolm-vla)
  and [public G1 datasets](https://huggingface.co/unitreerobotics/datasets)
  establish the 30 Hz imitation-learning contract used here. Their
  [G1 Dex1 Conveyor Sorting dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_ConveyorSorting)
  is a candidate dynamic-task auxiliary dataset, but it must pass action,
  camera, rate, frame, and normalization round-trip tests before any
  low-weight ablation.
