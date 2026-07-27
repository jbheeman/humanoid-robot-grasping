# VLA timing-alignment experiment ledger

Date: 2026-07-23

## Decision

Retain `pose23-latency-future1-t1-right9/steps_750_action_model.pt` as the
offline-selected checkpoint. It is an experiment artifact only and is not
authorized for robot execution.

## Data contract

- Immutable source snapshot:
  `plush_touch_canonical_block_v28/CANONICAL_MANIFEST.json`
- Real command-to-observed-state lag: 3 frames on train and validation.
- Isaac command-to-observed-state lag: 1 frame on train and validation.
- Corrected target semantics: state at `t + lookahead`, anchored at state `t`.
- Candidate lookaheads: 1 frame (33 ms) and 3 frames (100 ms) at 30 Hz.
- Terminal observations without a future label are dropped, never clamped.
- Relative-action statistics use train episodes only, with 75% real and 25%
  synthetic source mass.
- Validation selects models; the test split remains sealed.

Exact TFDS-to-HDF5 comparisons passed for sampled real and synthetic episodes
at both lookaheads. Relative Pose23 encode/reconstruct round-trip error was
`5.96e-8`.

## Validation results

Weighted metric is `0.75 * real ADE + 0.25 * synthetic ADE`.

| Variant | Weighted right-XYZ ADE |
|---|---:|
| Recorded-command relative, 750 steps | 0.08535 m |
| Achieved future1 relative, right9 | **0.06695 m** |
| Achieved future3 relative, right9 | 0.07132 m |
| Achieved future1 relative, XYZ-only | 0.06717 m |
| Achieved future1 relative, XYZ + 0.1 rotation | 0.06802 m |
| Achieved future1 absolute, right9 | 0.11189 m |
| Achieved future3 absolute, right9 | 0.11214 m |

The corrected future1 contract improves weighted ADE by 21.6% over the prior
relative pilot. Loss reweighting and preservation of the base model's absolute
action convention do not help.

## Remaining failure

The selected checkpoint does not beat the current-pose or per-horizon mean
baselines. Its raw predicted displacement is 2.14 times the target displacement
on real validation and 1.49 times on synthetic validation. Roughly 8.8% of
normalized outputs saturate.

A validation-tuned translation gain of 0.05 produces only a small improvement
over holding pose. That gain is diagnostic evidence, not a deployment
calibration, and must not authorize hardware motion.

## Next experiment

Build a non-paused, physics-qualified Isaac closed-loop evaluator that:

1. loads the selected future1 relative checkpoint and its exact statistics;
2. recomputes actions from fresh rendered observations;
3. consumes one bounded receding-horizon waypoint per control tick;
4. applies the geometric IK/table safety gateway independently of the model;
5. scores right-palm/rabbit collision as success;
6. compares raw output, conservative gain calibration, hold, and scripted IK;
7. requires at least 80% held-out block success and zero unsafe contacts.

Do not run more long imitation-training schedules until closed-loop failures
identify whether the next change belongs in data, action calibration, temporal
observation, or model capacity.
