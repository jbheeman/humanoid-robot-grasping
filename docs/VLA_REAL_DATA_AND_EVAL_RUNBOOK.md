# VLA real-data and closed-loop evaluation runbook

This workflow is offline. None of these commands imports ROS, connects to the
G1, or authorizes physical movement.

## Real demonstration contract

Record the head camera at its native 60 Hz if desired, but write causal policy
samples at 30 Hz. Each saved frame must contain a monotonic timestamp, one RGB
path, seven right-arm state values, and seven right-arm target values in the
same joint order and radians as the synthetic set.

Collect randomized blocks across five rabbit path angles, three speeds, and two
right-arm starts. `collection_block_id` identifies repeats that must stay in
one split. Use at least three independent blocks; a single recording session
cannot safely become train, validation, and test.

After visually checking the first true palm/rabbit contact frame:

```bash
uv run python scripts/training/label_real_plush_episode.py \
  /data/plush_touch/episode_0042 \
  --session 2026-07-23-afternoon \
  --block angle15-speed20-tucked-repeat1 \
  --path-angle-deg 15 \
  --speed-m-s 0.20 \
  --arm-start tucked \
  --visual-contact-frame 47
```

Audit without copying images:

```bash
uv run python scripts/training/audit_and_split_real_plush_episodes.py \
  /data/plush_touch \
  --output-dir runs/real_plush_qc/2026-07-23
```

The audit rejects missing frames, non-monotonic or non-30 Hz samples, invalid
seven-joint state/action vectors, fewer than 0.4 seconds before contact,
unreviewed/misaligned contact labels, missing motion-condition metadata, and
exact duplicate contact images.

- Fewer than 20 accepted episodes: pipeline validation only.
- 20–29 accepted episodes: grouped cross-validation pilot.
- 30 or more: group-disjoint train/validation/test manifests.
- Previously inspected test episodes must be audited with `--legacy`; they are
  never used for model selection.

## Timestamped VLA chunks

Each proposal records observation capture, synchronized state anchor, inference
start, inference completion, submission time, and derived per-waypoint times.
At 30 Hz with 0.37 seconds of capture-to-command delay, waypoints 0–10 have
expired and execution begins at waypoint 11. Later ticks skip any additional
expired commands. A fully expired chunk, stale observation, desynchronized
state, or out-of-order observation holds instead of replaying old motion.

## Closed-loop Isaac evidence

The Isaac producer writes one JSONL record per policy, random seed, and latency
profile. Required policies are `hold`, `ground_truth`, `scripted_ik`,
`tracker_cv_ik`, and `vla`. Required latency profiles are `p50`, `p95`, and
`1.5xp95`. Physics must continue advancing while inference delay is applied.

Evaluate a 120-seed pilot:

```bash
uv run python scripts/training/evaluate_vla_isaac_closed_loop.py \
  runs/isaac_rollouts/*.jsonl \
  --pilot \
  --output runs/isaac_rollouts/pilot_report.json
```

Remove `--pilot` for the final gate. The final VLA gate requires at least 300
rollouts, 80% physics-qualified block success, a 95% Wilson lower bound of at
least 70%, meaningful improvement over hold, zero prohibited contacts, zero
stale-waypoint executions, bounded safety intervention, and adequate results
at all three latency profiles. Ground-truth replay and scripted IK must first
prove the simulator/evaluator itself works.

A block is not a one-frame touch. It requires palm/rabbit contact, a sufficient
drop in rabbit speed or absolute stopping speed, bounded post-contact progress,
at least 0.2 seconds of dwell, bounded impact, and no prohibited collision.
Every report explicitly leaves `robot_execution_authorized` false.
