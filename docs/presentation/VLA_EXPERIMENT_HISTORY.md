---
title: Explain what the VLA experiments taught us
contentType: Conceptual
---

# Explain what the VLA experiments taught us

This page explains the complete Vision-Language-Action (VLA) experiment ladder on the dual-RTX 3090 host. It distinguishes training attempts, evaluations, infrastructure failures, and models that passed enough checks to reach the robot.

## What the run store contains

The inspected directory `/data1/aarav/data-stores/g1-bunny-vla/runs/unifolm_plush_touch` contains 38 top-level artifact directories:

- 36 training or smoke-test attempts
- Two merged-model selection directories
- 32 attempts that produced at least one action-model checkpoint
- Four attempts that stopped before producing an action-model checkpoint

Calling all 38 directories “successful training runs” would be inaccurate. Some are smoke tests, interrupted jobs, or recovery attempts. Together, they document the roughly 40-run research process.

## How to read the evaluation numbers

Average displacement error (ADE) measures the mean Euclidean distance between predicted and target right-hand positions over a trajectory. Lower values are better.

The campaign used a weighted ADE across real and simulated validation sources. It also checked:

- **Final displacement error (FDE)**: error at the final prediction step
- **Hold-pose baseline**: error from predicting that the hand does not move
- **Mean-action baseline**: error from predicting the average training action
- **Magnitude ratio**: predicted displacement divided by target displacement
- **Saturation**: fraction of normalized outputs at their clipping limit
- **Visual conditioning**: whether shuffling or occluding images changes predictions

A checkpoint could improve ADE and still fail promotion. It needed to beat trivial baselines, maintain reasonable action scale, react to visual input, and pass safety-oriented evaluation.

## The 36 training attempts

The run names record the experiment design. The table groups every training directory inspected on July 27, 2026.

| Experiment group | Count | Runs | Purpose and outcome |
| --- | ---: | --- | --- |
| Initial full training | 1 | `plush-touch-full` | Trained the first complete UniFoLM adaptation; selected step 8,500 |
| v28 staged schedule | 5 | `block-v28-smoke`, `block-v28-smoke2`, `block-v28-sim-3k`, `block-v28-mixed-4k`, `block-v28-real-1k` | Verified the loader, then trained on simulation, mixed data, and real data |
| Dynamic-action pilots | 4 | `dynamic-pilot-weighted-smoke`, `dynamic-pilot-t1-all23`, `dynamic-pilot-t1-right9`, `dynamic-pilot-t5s3-right9` | Compared all 23 actions, right-arm weighting, and image history |
| Pose23 representation pilots | 6 | `pose23-smoke-relative-t1-right9`, `pose23-smoke-relative-t5s3-right9`, `pose23-pilot-absolute-t1-right9`, `pose23-pilot-absolute-t5s3-right9`, `pose23-pilot-relative-t1-right9`, `pose23-pilot-relative-t5s3-right9` | Compared absolute pose targets with current-pose-anchored relative targets |
| Longer relative training | 2 | `pose23-relative-t1-fresh3000-seed1337`, `pose23-relative-t1-cont750-plus3000-lr1e5` | Tested whether more optimization or checkpoint continuation solved the error |
| Timing-aligned targets | 2 | `pose23-latency-future1-t1-right9`, `pose23-latency-future3-t1-right9` | Shifted labels to states achieved after one or three future frames |
| Translation weighting | 2 | `pose23-translation-future1-xyz3`, `pose23-translation-future1-xyz3-rot01` | Increased translation loss and reduced rotation influence |
| Absolute future targets | 2 | `pose23-future1-absolute-t1-right9`, `pose23-future3-absolute-t1-right9` | Tested whether future timing alone explained the gain |
| v29 motion-focused data | 2 | `v29-67real-motion-smoke`, `v29-67real-motion-mixed-4k` | Added 67 canonical real episodes and emphasized moving windows |
| v29 automated and adaptive campaign | 7 | `v29-67real-auto-20260723-attempt1`, three `gpu1safe` attempts, and adaptive attempts 20–22 | Added checkpoint evaluation, GPU recovery, seed changes, warm starts, and rejection gates |
| v30 active-horizon campaign | 3 | `v30-active-horizon-smoke-20260724-1103`, `v30-active-horizon-20260724-mixed_active`, `v30-active-horizon-20260724-mixed_active-gpu1-recovery` | Increased active-motion density; the full run and recovery failed before a promoted result |

The store also contains `selected_merged_vla` and `block-v28-selected-merged`. These are packaged selections, not new training attempts.

## Experiment 1: Establish a complete training path

The first `plush-touch-full` run proved that the UniFoLM base model could train on the custom G1 bunny task. The automated selector chose step 8,500.

The selected model produced approximately 0.130 m right-hand ADE on real samples and 0.150 m on simulation samples. This result established a working pipeline, but its trajectory accuracy was too weak for deployment.

## Experiment 2: Stage simulation, mixed, and real data

The v28 schedule separated training into:

1. A 3,000-step simulation stage
2. A 4,000-step mixed simulation-and-real stage
3. A 1,000-step real-only stage

The selected real-stage checkpoint achieved 0.1443 m on the sealed real test. It remained worse than the current-pose baseline. The result showed that adding a real-data finishing stage did not fix the target representation.

## Experiment 3: Restrict learning to the task-relevant arm

The dynamic pilots compared:

- One image and all 23 action dimensions
- One image with the right nine dimensions emphasized
- Five images, stride three, with the right nine dimensions emphasized

The one-frame right-arm variant scored 0.12464 m weighted ADE. It beat the other dynamic variants, but failed the baseline, bias, and saturation gates. Five-frame history added computation without improving the selected score.

## Experiment 4: Change from absolute to relative actions

The Pose23 pilots anchored predicted motion to the measured current pose. This representation reduced irrelevant global-pose variation.

| Representation | One-frame weighted ADE | Five-frame weighted ADE |
| --- | ---: | ---: |
| Absolute Pose23 | 0.12517 m | 0.13211 m |
| Anchored relative Pose23 | 0.08535 m | 0.09326 m |

The one-frame relative model improved 31.8% over the matched one-frame absolute model. The five-frame relative model also beat its absolute counterpart. One-frame input remained stronger than five-frame history.

## Experiment 5: Test whether longer training solves the remaining error

The team trained a fresh relative model for 3,000 steps and continued the 750-step model for another 3,000 steps at a lower learning rate.

| Run | Weighted ADE |
| --- | ---: |
| Original 750-step relative pilot | 0.08535 m |
| Continued model | 0.08583 m |
| Fresh 3,000-step model | 0.08886 m |

Neither longer run improved the pilot. The remaining problem was not insufficient optimization time.

## Experiment 6: Align labels with achieved future state

Robot commands and measured motion were offset by sensing, inference, transport, and actuation latency. The timing-alignment runs trained against the state achieved after the observation.

| Target | Weighted ADE |
| --- | ---: |
| Previous anchored-relative target | 0.08535 m |
| Three-frame future target | 0.07132 m |
| One-frame future target | 0.06695 m |

The one-frame future target improved 21.6% over the prior relative pilot. This was the strongest representation-level gain in the campaign.

The model still failed promotion. Its full-scale output did not beat the hold-pose baseline, and the selected report did not demonstrate visual conditioning. A translation gain sweep suggested that only a small fraction of the predicted action was useful.

## Experiment 7: Change loss weighting without changing semantics

The translation-focused runs tested a three-times translation weight and a version with rotation weighted at 0.1.

| Variant | Weighted ADE |
| --- | ---: |
| Translation ×3 | 0.06717 m |
| Translation ×3, rotation ×0.1 | 0.06802 m |

These runs stayed close to the one-frame timing result. Loss weighting did not produce a new breakthrough.

The absolute-future controls scored 0.11189 m and 0.11214 m. Their failure showed that future timing alone was insufficient. The model also needed the current-pose-relative action representation.

## Experiment 8: Add curated real data and automate rejection

The v29 dataset combined 480 Isaac Sim episodes with 67 canonical real episodes. The campaign imported 19 comparable historical evaluations, then ran three new adaptive attempts.

| Attempt | Configuration | Weighted ADE | Result |
| --- | --- | ---: | --- |
| GPU-safe attempt 1 | Earlier automated run | 0.06760 m | Rejected |
| GPU-safe attempt 2 | Earlier automated run | 0.07072 m | Rejected |
| Adaptive 20 | Seed 61, warm start, learning rate 1e-5 | 0.06799 m | Validation rejected |
| Adaptive 21 | Seed 62, fresh start, learning rate 5e-6 | 0.07995 m | Validation rejected |
| Adaptive 22 | Seed 63, warm start, learning rate 5e-6 | 0.06772 m | Validation rejected |

The controller stopped after ordinal 22 because the campaign had not improved enough to justify more attempts. No checkpoint received robot-execution authorization.

## Experiment 9: Increase active-motion density

The v30 data contract raised active training windows to 77.6%. Its smoke test completed, but the mixed active run failed during distributed training. A GPU 1 recovery attempt also failed, and the queue ended with `missing evaluation status`.

The v30 campaign produced no comparable promoted score. It should be presented as an infrastructure and evaluation failure, not as evidence that active-horizon data was worse.

## What the VLA work established

The campaign produced five defensible conclusions:

1. Relative actions fit this task better than absolute robot poses
2. One-frame observations outperformed the tested five-frame configuration
3. Achieved-future labels fit robot timing better than recorded command labels
4. Longer training and loss reweighting did not fix action scale or baseline performance
5. Offline ADE alone was not enough to authorize physical execution

The current demo branch does not include the training stack. It retains the deterministic perception, prediction, inverse kinematics, collision checking, and trajectory-generation path. The VLA work remains valuable as a documented research branch and as evidence for how future learned policies should represent time and actions.
