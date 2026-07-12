"""GPU-batched suspended G1 arm simulation using MuJoCo MJX."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from object_tracking.arm_tracking.joints import RIGHT_ARM_JOINT_NAMES
from object_tracking.g1_sim_cli import MODEL_REL, REVISION, TORQUE_LIMITS, root


def _load():
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    path = root() / MODEL_REL
    wrapper = path.parent / ".g1_gpu_suspended.xml"
    wrapper.write_text(
        '<mujoco><include file="g1_29dof.xml"/>'
        '<equality><weld name="hanger" body1="pelvis"/></equality></mujoco>'
    )
    model = mujoco.MjModel.from_xml_path(str(wrapper))
    # MJX 3.3 does not implement one visual mesh collision pair in this model.
    # Contacts are irrelevant while the pelvis is welded and only the arm moves.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    qpos, qvel, actuators = [], [], []
    for name in RIGHT_ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos.append(int(model.jnt_qposadr[jid]))
        qvel.append(int(model.jnt_dofadr[jid]))
        actuators.append(int(np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0]))
    device = jax.devices("gpu")[0]
    return jax, jnp, mjx, mjx.put_model(model, device=device), qpos, qvel, actuators


def doctor(_: argparse.Namespace) -> int:
    import jax

    gpu = jax.devices("gpu")[0]
    payload = {
        "ok": True,
        "backend": "mujoco-mjx",
        "device": str(gpu),
        "model_revision": REVISION,
        "right_arm_joints": list(RIGHT_ARM_JOINT_NAMES),
        "pelvis_welded": True,
        "contacts_enabled": False,
        "physical_robot_commands": False,
        "vram_policy": {
            "preallocate": os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "memory_fraction": os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def benchmark(args: argparse.Namespace) -> int:
    jax, jnp, mjx, model, qpos, qvel, actuators = _load()
    one = mjx.make_data(model)
    scenarios = 7 * 6
    environments = args.candidates * scenarios
    data = jax.tree.map(lambda x: jnp.repeat(x[None, ...], environments, axis=0), one)
    qpos_i, qvel_i, act_i = map(jnp.asarray, (qpos, qvel, actuators))
    candidate_kp = jnp.linspace(35.0, 80.0, args.candidates)
    candidate_kd = jnp.linspace(3.5, 0.8, args.candidates)
    kp = jnp.repeat(candidate_kp, scenarios)
    kd = jnp.repeat(candidate_kd, scenarios)
    amplitudes = jnp.asarray((-0.05, -0.03, -0.01, 0.01, 0.03, 0.05))
    requested = data.qpos[:, qpos_i]
    scenario_ids = jnp.tile(jnp.arange(scenarios), args.candidates)
    joints = scenario_ids // 6
    scenario_amplitudes = amplitudes[scenario_ids % 6]
    requested = requested.at[jnp.arange(environments), joints].add(scenario_amplitudes)
    torque_limits = jnp.asarray(TORQUE_LIMITS)

    def body(carry, _):
        d, error_sum = carry
        q = d.qpos[:, qpos_i]
        dq = d.qvel[:, qvel_i]
        error = requested - q
        tau = jnp.clip(kp[:, None] * error - kd[:, None] * dq, -torque_limits, torque_limits)
        d = d.replace(ctrl=d.ctrl.at[:, act_i].set(tau))
        d = jax.vmap(mjx.step, in_axes=(None, 0))(model, d)
        selected = error[jnp.arange(environments), joints]
        normalized = selected / jnp.abs(scenario_amplitudes)
        return (d, error_sum + normalized * normalized), None

    run = jax.jit(
        lambda d: jax.lax.scan(
            body, (d, jnp.zeros(environments)), None, length=args.steps
        )[0]
    )
    started = time.perf_counter()
    final, error_sum = run(data)
    jax.block_until_ready(final.qpos)
    elapsed = time.perf_counter() - started
    normalized_rmse = np.asarray(jnp.sqrt(error_sum / args.steps)).reshape(
        args.candidates, scenarios
    )
    candidate_scores = np.percentile(normalized_rmse, 95, axis=1)
    best_index = int(candidate_scores.argmin())
    result = {
        "schema_version": 1,
        "backend": "mujoco-mjx",
        "device": str(jax.devices("gpu")[0]),
        "candidates": args.candidates,
        "scenarios_per_candidate": scenarios,
        "environments": environments,
        "steps_per_environment": args.steps,
        "simulated_steps": environments * args.steps,
        "wall_seconds_including_compile": elapsed,
        "steps_per_second": environments * args.steps / elapsed,
        "best_normalized_p95_rmse": float(candidate_scores[best_index]),
        "best_candidate": {
            "kp": float(candidate_kp[best_index]),
            "kd": float(candidate_kd[best_index]),
        },
        "best_scenario_normalized_rmse": normalized_rmse[best_index].tolist(),
        "median_candidate_score": float(np.median(candidate_scores)),
        "real_robot_validated": False,
        "contacts_enabled": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = root() / "runs/simulation-gpu/latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.set_defaults(func=doctor)
    b = sub.add_parser("benchmark")
    b.add_argument("--candidates", type=int, default=32)
    b.add_argument("--steps", type=int, default=1500)
    b.set_defaults(func=benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
