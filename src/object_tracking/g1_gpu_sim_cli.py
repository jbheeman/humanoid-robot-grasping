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
from object_tracking.g1_exchange import merge_elites


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
    rng = np.random.default_rng(args.seed)
    bounds = {
        "kp": (30.0, 100.0),
        "kd": (0.5, 5.0),
        "vmax": (0.05, 0.30),
        "amax": (0.25, 3.0),
    }
    values = {
        name: rng.uniform(lower, upper, args.candidates)
        for name, (lower, upper) in bounds.items()
    }
    if args.checkpoint and Path(args.checkpoint).is_file():
        checkpoint = json.loads(Path(args.checkpoint).read_text())
        elites = checkpoint.get("elite_candidates") or [checkpoint.get("best_candidate", {})]
    else:
        elites = []
    if args.exchange and Path(args.exchange).is_file():
        try:
            exchange = json.loads(Path(args.exchange).read_text()).get("candidates", [])
        except json.JSONDecodeError:
            exchange = []
        elites = merge_elites(elites, exchange)
    if elites:
        local_count = args.candidates // 2
        elite_indices = rng.integers(0, len(elites), local_count)
        for name, (lower, upper) in bounds.items():
            centers = np.asarray(
                [elites[index].get(name, (lower + upper) / 2) for index in elite_indices]
            )
            if len(centers):
                values[name][-local_count:] = np.clip(
                    rng.normal(centers, (upper - lower) * 0.06, local_count),
                    lower,
                    upper,
                )
    candidate_kp = jnp.asarray(values["kp"])
    candidate_kd = jnp.asarray(values["kd"])
    candidate_vmax = jnp.asarray(values["vmax"])
    candidate_amax = jnp.asarray(values["amax"])
    kp = jnp.repeat(candidate_kp, scenarios)
    kd = jnp.repeat(candidate_kd, scenarios)
    vmax = jnp.repeat(candidate_vmax, scenarios)
    amax = jnp.repeat(candidate_amax, scenarios)
    amplitudes = jnp.asarray((-0.05, -0.03, -0.01, 0.01, 0.03, 0.05))
    requested = data.qpos[:, qpos_i]
    scenario_ids = jnp.tile(jnp.arange(scenarios), args.candidates)
    joints = scenario_ids // 6
    scenario_amplitudes = amplitudes[scenario_ids % 6]
    requested = requested.at[jnp.arange(environments), joints].add(scenario_amplitudes)
    torque_limits = jnp.asarray(TORQUE_LIMITS)

    baseline = data.qpos[:, qpos_i]
    dt = 0.002

    def body(carry, _):
        d, error_sum, target, command_velocity = carry
        q = d.qpos[:, qpos_i]
        dq = d.qvel[:, qvel_i]
        desired_velocity = jnp.clip((requested - target) / dt, -vmax[:, None], vmax[:, None])
        command_velocity += jnp.clip(
            desired_velocity - command_velocity,
            -amax[:, None] * dt,
            amax[:, None] * dt,
        )
        target += command_velocity * dt
        tracking_error = target - q
        tau = jnp.clip(
            kp[:, None] * tracking_error - kd[:, None] * dq,
            -torque_limits,
            torque_limits,
        )
        d = d.replace(ctrl=d.ctrl.at[:, act_i].set(tau))
        d = jax.vmap(mjx.step, in_axes=(None, 0))(model, d)
        final_error = requested - d.qpos[:, qpos_i]
        selected = final_error[jnp.arange(environments), joints]
        normalized = selected / jnp.abs(scenario_amplitudes)
        return (d, error_sum + normalized * normalized, target, command_velocity), None

    run = jax.jit(
        lambda d: jax.lax.scan(
            body,
            (
                d,
                jnp.zeros(environments),
                baseline,
                jnp.zeros((environments, 7)),
            ),
            None,
            length=args.steps,
        )[0]
    )
    started = time.perf_counter()
    final, error_sum, _, _ = run(data)
    jax.block_until_ready(final.qpos)
    elapsed = time.perf_counter() - started
    normalized_rmse = np.asarray(jnp.sqrt(error_sum / args.steps)).reshape(
        args.candidates, scenarios
    )
    candidate_scores = np.percentile(normalized_rmse, 95, axis=1)
    best_index = int(candidate_scores.argmin())
    elite_indices = np.argsort(candidate_scores)[: min(16, args.candidates)]

    def candidate_payload(index: int) -> dict[str, float]:
        return {
            "kp": float(candidate_kp[index]),
            "kd": float(candidate_kd[index]),
            "vmax": float(candidate_vmax[index]),
            "amax": float(candidate_amax[index]),
            "score": float(candidate_scores[index]),
        }

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
        "best_candidate": candidate_payload(best_index),
        "elite_candidates": [candidate_payload(int(index)) for index in elite_indices],
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
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--checkpoint")
    b.add_argument("--exchange", help="local compact exchange JSON")
    b.set_defaults(func=benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
