---
title: Present the experiments and results
contentType: Reference
---

# Present the experiments and results

This page summarizes the strongest quantitative results from the dual-3090 Isaac Sim and UniFoLM training store. All values come from manifests or evaluation reports under `/data1/aarav/data-stores/g1-bunny-vla`.

## Training infrastructure and artifact scale

The `/data1/aarav` store contained about 754 GB when inspected on July 27, 2026:

| Category | Size | Contents |
| --- | ---: | --- |
| Runs | 695 GB | Training states, checkpoints, diagnostics, and campaign ledgers |
| Models | 34 GB | UniFoLM-VLA and vision-language base models |
| Datasets | 26 GB | Canonical HDF5, RLDS/TFDS, raw real data, and Isaac Sim variants |
| Worktrees | 3.5 MB | Lightweight evaluation code and simulation baselines |

The run store is large because a single distributed training state can exceed 40 GB. Portable action-head checkpoints are about 1.2 GB.

## Dataset progression

The data pipeline grew in three major steps:

| Dataset | Isaac episodes | Real episodes | Total frames | Purpose |
| --- | ---: | ---: | ---: | --- |
| `plush_touch_canonical_v1` | 480 | 15 | 42,374 | First combined static-touch dataset |
| `plush_touch_canonical_block_v28` | 480 | 15 | 28,835 | Moving-bunny simulation with original small real set |
| `plush_touch_canonical_v29_67real` | 480 | 67 | 32,051 | Moving-bunny simulation with expanded curated real set |

The v29 dataset uses:

- 389 sim and 46 real training episodes
- 41 sim and 12 real validation episodes
- 50 sim and 9 real sealed test episodes

Real-data curation reduced 127 audited episodes to 97 accepted/reviewed episodes, then to 67 episodes that passed the canonical timing, motion, image, and contact contract.

## Isaac Sim data-generation work

The moving-bunny generator went through more than 20 named variants before v28. The iterations addressed:

- Bunny scale, contact geometry, friction, and spin
- Camera realism, shadows, antialiasing, and occlusion
- Direct versus indirect contact paths
- Left-side spawn and visibility failures
- Temporal-history reset and post-contact capture

The final `moving_block_480_v28` dataset occupies about 4.6 GB and contains 480 synthetic episodes.

## Controller tuning in simulation

An earlier six-hour simulation campaign evaluated 9,312 candidate trials with four randomized replays per candidate.

| Metric | Best simulated result |
| --- | ---: |
| Evaluation time | 6.00 hours |
| Candidate trials evaluated | 9,312 |
| Failure rate | 2.38% |
| Overall p95 joint error | 0.0517 rad |
| Shoulder/elbow p95 error | 0.0437 rad |
| Selected `kp` | 110.0 |
| Selected `kd` | 4.50 |
| Selected velocity limit | 0.259 rad/s |
| Selected acceleration limit | 3.93 rad/s² |

These gains were not marked as real-robot validated. They are simulation evidence, not hardware defaults.

## VLA experiment ladder

The UniFoLM run directory contains 36 train or smoke attempts, of which 32 produced action-model checkpoints. The sections below summarize the experiment ladder. [The VLA experiment history](VLA_EXPERIMENT_HISTORY.md) inventories every run group and explains each comparison.

### Initial 10,000-step training

The first full schedule selected step 8,500:

| Source | Right-XYZ ADE | Right-XYZ FDE |
| --- | ---: | ---: |
| Real validation | 0.1300 m | 0.1446 m |
| Sim validation | 0.1504 m | 0.1446 m |

This run completed and produced a merged model, but it did not establish closed-loop success.

### Moving-bunny v28 staged training

The v28 pipeline trained sequentially on simulation, mixed data, and real data. It selected the real-stage step-250 checkpoint.

| Split/source | Right-XYZ ADE | Right-XYZ FDE |
| --- | ---: | ---: |
| Real validation | 0.1257 m | 0.1432 m |
| Sim validation | 0.1378 m | 0.1096 m |
| Real sealed test | 0.1443 m | 0.1429 m |
| Sim sealed test | 0.1453 m | 0.1302 m |

The offline gate required real validation ADE below 0.08 m and FDE below 0.10 m. The checkpoint failed both limits and was not authorized for robot execution.

### Temporal-history pilots

Adding five observations over 0.4 seconds did not improve the original absolute-action model:

| Variant | Weighted validation ADE |
| --- | ---: |
| Single frame, all 23 actions | 0.12684 m |
| Single frame, right arm | 0.12464 m |
| Five frames, right arm | 0.12853 m |

All three variants were worse than the current-pose baseline.

### Relative-action representation

Anchored relative actions improved the learned metric at matched budgets:

| Variant | Weighted validation ADE |
| --- | ---: |
| Absolute, single frame | 0.12517 m |
| Relative, single frame | 0.08535 m |
| Absolute, five frames | 0.13211 m |
| Relative, five frames | 0.09326 m |

Relative single-frame actions improved the weighted error by about 31.8% over the matched absolute pilot. The model still failed the baseline gate because holding or averaging pose remained more accurate.

### Future-state target alignment

The most important experiment aligned the target with the state achieved after measured command latency:

| Target contract | Weighted right-XYZ ADE |
| --- | ---: |
| Prior recorded-command relative target | 0.08535 m |
| Achieved future state, 1 frame | **0.06695 m** |
| Achieved future state, 3 frames | 0.07132 m |
| Achieved future state, XYZ only | 0.06717 m |
| Achieved future state, XYZ plus 0.1 rotation | 0.06802 m |
| Absolute future state, 1 frame | 0.11189 m |
| Absolute future state, 3 frames | 0.11214 m |

The one-frame achieved-future target improved the main metric by 21.6% over the prior relative pilot.

The selected model still had three critical problems:

- It did not beat the current-pose or per-horizon mean baselines
- Predicted displacement was 2.14 times the target on real validation
- About 8.8% of normalized outputs saturated

This is a meaningful research improvement, not a deployable policy.

### Active-motion analysis and v30

The v30 audit separated windows with more than 0.01 m of target motion:

| Source | Windows | Active windows | Active fraction |
| --- | ---: | ---: | ---: |
| Real training | 3,186 | 2,473 | 77.6% |
| Sim training | 22,176 | 14,711 | 66.3% |

The active-window experiment was designed to reduce the dominance of nearly static targets. Its distributed run failed after GPU/NCCL instability, a recovery attempt, and an evaluation argument error. It produced no promoted result.

### Adaptive v29 campaign

The campaign controller ran 22 attempts across learning rates, warmup schedules, real-data weights, augmentation, seeds, and warm starts.

| Campaign result | Value |
| --- | ---: |
| Attempts | 22 |
| Best imported weighted ADE | 0.06760 m |
| Latest attempt weighted ADE | 0.06772 m |
| Final phase | No-progress pause |
| Robot execution authorized | No |

The last three attempts all remained visually conditioned, but failed baseline, action-magnitude, or output-saturation gates. The campaign correctly stopped instead of selecting a cosmetically better but unsafe checkpoint.

## Perception and IK performance

The deployment work improved both perception and geometric control:

| Component | Result |
| --- | ---: |
| Representative July 27 camera rate | About 57 FPS |
| Representative July 27 YOLO rate | About 49–51 FPS |
| Analytic IK median local step | 0.080 ms |
| Finite-difference IK median local step | 0.136 ms |
| Analytic step latency reduction | About 41% |
| Accepted analytic benchmark steps | 64/64 |
| Mean position error after one analytic step | 0.30 mm |

The perception rates came from dry-run robot/GB10 sessions. The IK values came from an offline canonical-model benchmark. The faster analytic IK has not yet completed the final physical bunny-tracking retest.

## What the negative results taught us

The main lesson is that more training was not the correct first response:

- A model can be visually conditioned and still predict the wrong motion magnitude
- Lower validation loss does not guarantee improvement over a hold-pose baseline
- Recorded command targets can be wrong when the observed state lags execution
- Temporal history does not help if action semantics and normalization remain incorrect
- Strict promotion gates prevented offline improvements from becoming unsafe hardware experiments
