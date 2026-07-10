# G1 Joint Audit and Dry-Run Tuning

The tuning workflow is deliberately split into two parts:

1. prove that the robot, URDF, IK, HTTP payload, and SDK use the same joint contract;
2. optimize perception from repeatable dry-run data without weakening movement safety gates.

Neither a high diagnostic score nor a successful dry run authorizes arm movement.

## Joint contract

The seven-element `right_arm_q` vector is:

| Vector | SDK2 slot | Joint |
|---:|---:|---|
| 0 | 22 | right shoulder pitch |
| 1 | 23 | right shoulder roll |
| 2 | 24 | right shoulder yaw |
| 3 | 25 | right elbow |
| 4 | 26 | right wrist roll |
| 5 | 27 | right wrist pitch |
| 6 | 28 | right wrist yaw |

This is the Unitree 29-DOF mapping. Slots 27 and 28 are not valid on a 23-DOF G1, so first confirm the physical robot variant. The bridge holds the left arm from LowState and publishes all 14 arm slots; it never publishes a partial right-arm command.

After fetching the pinned assets, audit URDF names, order, and limits:

```bash
./scripts/fetch_unitree_arm_assets.sh
uv run g1-tune joint-audit
```

The GB10 setup now performs this audit automatically. IK also refuses to initialize unless Pinocchio reduces the model to exactly seven coordinates in the order above.

The currently locked `pin` installation does not expose `pinocchio.casadi`; therefore a static URDF audit can pass while IK construction still fails. Treat `ik_unavailable` as a dependency/runtime blocker, not as a parameter to tune. Resolve the P1 IK task in `docs/review.md` before any physical command test.

This static audit is necessary but not sufficient. Before motion, capture LowState on the physical robot, confirm it is the 29-DOF model, compare all seven reported values with the robot's known pose, and perform the supervised `0.05 rad` smoke test described in the arm runbook.

## Record a labeled baseline

Start in dry-run with a short label and notes:

```bash
RESEARCH_LABEL=baseline_static \
RESEARCH_NOTES="plushie 0.8m away, room light, table center" \
RESEARCH_HZ=15 \
./scripts/run_gb10_vision_server.sh
```

Use the same scene sequence for every comparison:

1. object stationary for 20 seconds;
2. slow horizontal motion for 20 seconds;
3. slow depth motion for 20 seconds;
4. partial occlusion for 10 seconds;
5. object absent for 10 seconds.

Analyze the newest session under the research root:

```bash
uv run g1-tune analyze runs/research/arm_tracking
```

The report explains capture rate, inference rate, depth age, RGB/depth skew, detection/track coverage, ID switches, dominant rejection reasons, and the next experiment to run. Its score is an observability/readiness diagnostic, not a safety score.

## Controlled parameter sweeps

Change one parameter family at a time. Give each run a label and keep the physical scene sequence identical.

Inference-size sweep:

```bash
RESEARCH_LABEL=imgsz_640 IMGSZ=640 ./scripts/run_gb10_vision_server.sh
RESEARCH_LABEL=imgsz_800 IMGSZ=800 ./scripts/run_gb10_vision_server.sh
RESEARCH_LABEL=imgsz_960 IMGSZ=960 ./scripts/run_gb10_vision_server.sh
```

Detector-confidence sweep:

```bash
RESEARCH_LABEL=conf_025 CONF=0.25 ./scripts/run_gb10_vision_server.sh
RESEARCH_LABEL=conf_035 CONF=0.35 ./scripts/run_gb10_vision_server.sh
RESEARCH_LABEL=conf_050 CONF=0.50 ./scripts/run_gb10_vision_server.sh
```

Compare every recorded session in a directory:

```bash
uv run g1-tune compare runs/research/arm_tracking
```

Only use that ranking when the runs used the same labeled scene. Detection coverage alone cannot measure accuracy: manually label a representative frame subset and select confidence/input size from precision, recall, latency, and track stability together.

## What may be tuned automatically

Safe offline automation can select among recorded perception profiles using labeled replay:

- YOLO input size and confidence;
- inference cadence;
- tracker match distance and missed-frame retention;
- depth ROI fraction, valid support, MAD, and clustering thresholds;
- alpha-beta filter gains and prediction horizon;
- support-plane RANSAC thresholds.

The current telemetry analyzer identifies which family to sweep. Full numerical auto-tuning requires synchronized raw RGB/Z16 replay plus ground-truth labels; structured JSONL alone is not enough to rescore different detector or depth parameters.

These values must not be automatically loosened from live success rate:

- joint limits and margin;
- maximum joint velocity/acceleration;
- following-error limit;
- target TTL and deadman;
- workspace bounds;
- calibration residual gates;
- collision and support-plane clearance;
- controller-ownership and standing gates.

Those are safety requirements, not optimization variables.
