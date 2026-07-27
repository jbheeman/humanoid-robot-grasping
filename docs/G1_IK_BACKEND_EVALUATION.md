# G1 IK backend evaluation

This evaluation is offline-only. It does not open ROS, DDS, a camera, or a
robot transport.

## Decision

Keep the pinned official Unitree XR-Teleoperate URDF and the existing
Pinocchio safety layer. Use Pinocchio's analytic, world-aligned frame Jacobian
for realtime translation servoing. Do not make
[`YuehChuan/unitreeG1_ik`](https://github.com/YuehChuan/unitreeG1_ik) the
production global solver.

The requested repository is useful as a compact damped-least-squares
reference, but it currently has no redistribution license, no convergence
result, no joint-limit enforcement, no self/table collision checks, and no
swept-path validation. Its bundled model also gives the right shoulder-roll
joint the mirrored left-arm range (`[-1.5882, 2.2515]`) instead of the official
right-arm range (`[-2.2515, 1.5882]`) and omits the official shoulder-roll
frame rotation.

Our production solver continues to use the pinned Apache-2.0
[`unitreerobotics/xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate)
model and IK reference at revision
`7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`.

## Reproducible benchmark

Run:

```bash
uv sync --group arm
uv run --group arm scripts/dev/benchmark_g1_ik_backends.py --samples 256
```

To compare a separately cloned YuehChuan checkout without copying its
unlicensed source into this repository:

```bash
uv run --group arm --with mujoco==3.2.4 \
  scripts/dev/benchmark_g1_ik_backends.py \
  --samples 64 \
  --yueh-checkout /path/to/unitreeG1_ik \
  --output artifacts/ik_benchmarks/g1_ik_backends.json
```

Every result is judged using the same canonical Unitree/XR Pinocchio FK,
joint limits, and collision model. The initial 64-target comparison on
2026-07-24 produced:

| Backend | Canonical success | Mean position error | Median latency | Unsafe outputs |
| --- | ---: | ---: | ---: | ---: |
| Existing bounded SciPy global IK | 64/64 | 1.13 mm | 230.3 ms | 0 |
| YuehChuan MuJoCo DLS reference | 0/64 | 9.53 mm | 15.2 ms | 0 in this small interior set |
| New Pinocchio analytic local DLS | 58/64 at ≤5 mm and ≤3°; 64/64 accepted | 0.30 mm after one step | 0.080 ms | 0 |
| Previous finite-difference local DLS | 58/64 at ≤5 mm and ≤3°; 64/64 accepted | 0.30 mm after one step | 0.136 ms | 0 |

The local benchmark measures a single bounded servo step, so success is
expected to converge over subsequent receding-horizon steps. Analytic and
finite-difference versions produced effectively identical error reduction
(96.76%), while the analytic median latency was about 41% lower.

## Runtime contract

The analytic DLS result is only a candidate. It still passes:

- canonical G1 joint limits with safety margins;
- maximum per-step joint discontinuity;
- full-link self-collision checks;
- table support-region clearance;
- dense swept-edge validation;
- the ROS deadman and downstream command guards.

The previous finite-difference Jacobian remains selectable through
`jacobian_backend="finite_difference"` for regression testing.

## Other libraries considered

- [Pink](https://github.com/stephane-caron/pink) is an actively maintained,
  Apache-2.0 Pinocchio QP solver with configuration and velocity limits.
- [Mink](https://github.com/kevinzakka/mink) is an actively maintained,
  Apache-2.0 MuJoCo QP solver with G1 examples and collision constraints.
- [Ruckig](https://github.com/pantor/ruckig) is appropriate after IK for
  velocity/acceleration/jerk-limited time parameterization; it is not an IK
  replacement.

Adding Pink or Mink to the robot runtime is not justified by this benchmark:
the analytic Pinocchio step already removes the finite-difference overhead
without adding a second robot model or another optimization dependency.
