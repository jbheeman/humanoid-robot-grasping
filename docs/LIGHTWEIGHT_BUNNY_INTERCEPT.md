# Lightweight moving-bunny interception

The default counter-test is a bounded controller, not a second end-to-end
policy:

```text
RGB-D / oracle object state
  -> timestamped 3D track
  -> CV, alpha-beta, or compact GRU forecast
  -> deadline/reachability-aware plane intercept
  -> full-pose or local analytic guarded IK
  -> Isaac position controller and contact physics
```

This experiment never authorizes physical robot motion. A human must separately
commission any promoted controller.

## Required staged gate

Use the exact same scenario IDs for every policy. The pilot design is 30
scenarios: five headings, three speeds, and two right-arm starts.

1. Run `hold` and `oracle_ik` with zero injected latency. Isaac physics must
   advance while joint targets move through the real actuator controller; do
   not teleport the palm. Require at least 29/30 qualified blocks and no
   prohibited contact before testing perception or a learned predictor.
2. Run `cv_ik`, `alpha_beta_ik`, and `gru_ik` at 100, 200, and 400 ms injected
   observation latency on the same scenarios. A missed deadline must become a
   hold/no-action result labelled `deadline_unreachable`; never chase a stale
   waypoint.
3. Only after the oracle and predictor stages pass, replace oracle positions
   with registered rendered RGB-D detections on the same scenario IDs.
4. Compare the winner with UniFoLM as a benchmark. Do not label a classical
   controller as `vla`.

Every rollout record must include:

- `policy`, numeric `latency_ms`, and stable `scenario_id`
- physics-qualified block fields used by `physics_qualified_success`
- 3D prediction error, plan failure reason, and IK rejection
- right-palm/bunny contact versus prohibited forearm/table/self contacts
- joint-limit and torque saturation counts

Aggregate results with:

```bash
uv run python scripts/training/evaluate_intercept_closed_loop.py \
  artifacts/intercept/rollouts.jsonl \
  --candidate alpha_beta_ik \
  --output artifacts/intercept/report.json
```

The palm target is not the bunny centre. `intercept_planner.py` offsets the palm
by the bunny contact radius and palm half-thickness and returns a normal facing
the oncoming bunny. If the current wrist orientation differs too much, use the
full-pose guarded IK solve before translation-only servoing.

The final promotion target is at least 80% physics-qualified blocks in every
100/200/400 ms slice, zero prohibited contacts or actuator saturation, and a
Wilson 95% lower bound of at least 70%. At least 50 scenarios per slice is
recommended for a final decision; 30 is only a pilot.
