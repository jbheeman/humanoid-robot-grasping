---
title: Trace presentation claims to evidence
contentType: Reference
---

# Trace presentation claims to evidence

This index records the evidence inspected on July 27, 2026. Remote files were read over SSH without modification.

## Host inventory

| Role | Host | Active project path | Observed state |
| --- | --- | --- | --- |
| Dual-3090 Isaac/training host | `128.114.152.136:6023` | `/home/aarav/Documents/g1-bunny-vla-workspace` | Experiment worktree ahead of its remote with substantial untracked artifacts |
| Training artifact store | Dual-3090 host | `/data1/aarav/data-stores/g1-bunny-vla` | 754 GB store containing datasets, models, runs, and worktrees |
| GB10 runtime | `192.168.0.66` | `/home/aarav/Documents/project` | Branch `aarav-vla`, commit `2317771`; dry-run perception active during inspection |
| Unitree G1 | `192.168.0.213` | `/home/unitree/humanoid-robot-grasping` | Branch `aarav-vla`, commit `f70ae89`; disarmed observation/control bridge active during inspection |

The host paths differ intentionally. Do not assume that all checkouts have identical commits or clean working trees.

## Git history coverage

The repository history was refreshed and inspected across every remote branch visible on July 27, 2026:

- `main`
- `advay`
- `aarav`
- `aarav-vla`
- `calibration`
- `demo-realtime-intercept-minimal`
- `neel`
- `neel2`
- `oliver`
- `sim`
- `vla-sim`
- `lightweight-intercept-isaac`

The cached `isaac/vla-sim` ref was also inspected. It is identical to `origin/vla-sim` and adds no unique commits.

The [implementation history](IMPLEMENTATION_HISTORY.md) uses commit dates, merge bases, branch-only commits, and file-level diffs to separate the hardware-control, simulation, VLA, calibration, and evaluation lines of work. Branch labels describe where work is most visible in Git; they do not imply exclusive individual ownership.

## Claim-to-source map

| Claim | Evidence |
| --- | --- |
| Training store is 754 GB | `du` of `/data1/aarav/data-stores/g1-bunny-vla` |
| Store contains 695 GB runs, 34 GB models, 26 GB datasets | Top-level `du` of the training store |
| v29 contains 547 episodes and 32,051 frames | `datasets/plush_touch_canonical_v29_67real/CANONICAL_MANIFEST.json` |
| v29 contains 480 sim and 67 real episodes | Same canonical manifest, grouped by `source` |
| 127 raw real episodes were audited | GB10 `runs/real_plush_qc/20260723_complete/summary.json` |
| 97 real episodes were accepted/reviewed for materialization | `datasets/real_plush_20260723_curated_v2/CURATED_MANIFEST.json` |
| 67 real episodes passed the canonical contract | `datasets/plush_touch_real_20260723_canonical_v2/CANONICAL_MANIFEST.json` |
| Six-hour controller search evaluated 9,312 trials | `worktrees/lightweight-intercept/simulation/baselines/g1_right_arm/validation_summary.json` |
| First full VLA selected step 8,500 | `runs/unifolm_plush_touch/AUTO_TRAINING_SUMMARY.json` |
| v28 sealed real test ADE was 0.1443 m | `runs/unifolm_plush_touch/block_v28_selected_test.json` |
| Relative single-frame pilot scored 0.08535 m | `runs/diagnostics/pose23_pilots/SELECTION.json` |
| Achieved-future single-frame pilot scored 0.06695 m | `runs/diagnostics/pose23_latency/SELECTION.json` |
| Future-state target improved error by 21.6% | Comparison recorded in `docs/VLA_TIMING_ALIGNMENT_RESULTS.md` |
| v29 adaptive campaign stopped after 22 attempts | `runs/automation/v29-67real-adaptive/CAMPAIGN_STATUS.json` |
| v30 failed without a promoted result | `runs/automation/v30-active-horizon-20260724/QUEUE_STATUS.json` and its queue log |
| Real training windows were 77.6% active | `runs/diagnostics/v30_motion_distribution.json` |
| VLA store contains 36 train or smoke attempts | Top-level directory inventory under `runs/unifolm_plush_touch`, excluding two merged-selection directories |
| 32 VLA attempts produced action-model checkpoints | Checkpoint-file inventory under each training directory |
| Analytic IK median step was 0.080 ms | `docs/G1_IK_BACKEND_EVALUATION.md` |
| July 27 camera and YOLO rates were about 57 and 49–51 FPS | GB10 `runs/research/arm_tracking/20260727_*/summary.json` |
| Physical IK tracking worked but was slow | Team-confirmed result supplied during documentation review |
| Faster analytic path awaits final physical retest | Team-confirmed current status plus repository history through commit `2317771` |
| Implementation evolved through distinct hardware, simulation, VLA, and oracle-evaluation lines | Complete fetched `origin/*` history summarized in `IMPLEMENTATION_HISTORY.md` |
| Current focused branch removed 25,756 lines across 183 files | Git diff from `c5c3366` to `d0a873f` |
| Current live test uses the high-FPS camera service and eight-sample readiness gate | Commit `97b9853` |
| Current branch retains jerk-limited Ruckig execution | Commits `c982321`, `90f56f0`, and `c5c3366`, plus `src/object_tracking/arm_tracking/arm_bridge.py` |

## Confidence labels

- **Artifact-verified:** A numerical value appears in a manifest, report, or run summary
- **Source-verified:** The implementation exists in the current repository
- **Team-confirmed:** The team observed the physical behavior, but no standardized aggregate metric exists
- **Pending:** The implementation exists, but the corresponding final hardware test has not occurred

The presentation uses these labels implicitly. Preserve the distinction when adding new results.
