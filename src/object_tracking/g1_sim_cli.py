"""Suspended G1 right-arm MuJoCo tuning.

Simulation output is advisory: candidates are never written into robot configuration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np

from object_tracking.arm_tracking.joints import RIGHT_ARM_JOINT_NAMES

REVISION = "ae6a8403e272733e9996ef59990880330496177f"
MODEL_REL = Path(".deps/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
OUTPUT_REL = Path("runs/simulation")
TORQUE_LIMITS = (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0)


def root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Candidate:
    kp: float
    kd: float
    vmax: float
    amax: float


def _wrapper_xml(model: Path) -> str:
    # The official file is a composed MJCF (many meshes), not one merged STL.
    return (
        '<mujoco model="g1_suspended"><include file="'
        + str(model.resolve())
        + '"/><equality><weld name="hanger" body1="pelvis"/></equality></mujoco>'
    )


def _load() -> tuple[Any, Any, list[int], list[int], list[int]]:
    import mujoco

    path = root() / MODEL_REL
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/sim/setup.sh")
    wrapper = path.parent / f".g1_suspended_{os.getpid()}.xml"
    wrapper.write_text(
        '<mujoco model="g1_suspended"><include file="g1_29dof.xml"/>'
        '<equality><weld name="hanger" body1="pelvis"/></equality></mujoco>'
    )
    model = mujoco.MjModel.from_xml_path(str(wrapper))
    data = mujoco.MjData(model)
    joint_ids, qpos, qvel = [], [], []
    for name in RIGHT_ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"official model has no joint {name}")
        joint_ids.append(jid)
        qpos.append(int(model.jnt_qposadr[jid]))
        qvel.append(int(model.jnt_dofadr[jid]))
    actuators = []
    for jid in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        if len(matches) != 1:
            raise RuntimeError(f"joint id {jid} has {len(matches)} actuators")
        actuators.append(int(matches[0]))
    return model, data, qpos, qvel, actuators


def _trial(payload: tuple[dict[str, float], int, float, int]) -> dict[str, Any]:
    import mujoco

    raw, joint_index, amplitude, seed = payload
    c = Candidate(**raw)
    model, data, qpos, qvel, actuators = _load()
    rng = random.Random(seed)
    model.opt.gravity[2] *= rng.uniform(0.98, 1.02)
    model.dof_damping[:] *= rng.uniform(0.85, 1.15)
    mujoco.mj_forward(model, data)
    baseline = data.qpos[qpos].copy()
    requested = baseline.copy()
    requested[joint_index] += amplitude
    target = baseline.copy()
    velocity = np.zeros(7)
    dt = float(model.opt.timestep)
    errors, torques, velocities, drift = [], [], [], []
    settled_at = None
    duration = 3.0
    for step in range(math.ceil(duration / dt)):
        if step % max(1, round(0.004 / dt)) == 0:
            delta = requested - target
            desired_v = np.clip(delta / 0.004, -c.vmax, c.vmax)
            velocity += np.clip(desired_v - velocity, -c.amax * 0.004, c.amax * 0.004)
            target += velocity * 0.004
        q = data.qpos[qpos]
        dq = data.qvel[qvel]
        tau = c.kp * (target - q) - c.kd * dq
        tau = np.clip(tau, -np.asarray(TORQUE_LIMITS), np.asarray(TORQUE_LIMITS))
        data.ctrl[actuators] = tau
        mujoco.mj_step(model, data)
        err = requested - data.qpos[qpos]
        errors.append(float(err[joint_index]))
        torques.append(float(abs(tau[joint_index])))
        velocities.append(float(data.qvel[qvel[joint_index]]))
        drift.append(float(np.max(np.abs(np.delete(data.qpos[qpos] - baseline, joint_index)))))
        if (
            settled_at is None
            and step * dt > 0.25
            and abs(err[joint_index]) < 0.01
            and abs(velocities[-1]) < 0.02
        ):
            settled_at = step * dt
        if not np.isfinite(data.qpos).all():
            return {"ok": False, "failure": "nonfinite"}
    e = np.asarray(errors)
    overshoot = max(0.0, float(np.max(np.sign(amplitude) * (amplitude - e)) - abs(amplitude)))
    result = {
        "ok": settled_at is not None and max(drift) < 0.08 and max(torques) <= TORQUE_LIMITS[joint_index],
        "joint": RIGHT_ARM_JOINT_NAMES[joint_index],
        "amplitude": amplitude,
        "rmse": float(np.sqrt(np.mean(e * e))),
        "p95_error": float(np.percentile(np.abs(e), 95)),
        "overshoot": overshoot,
        "settle_s": settled_at,
        "peak_torque": max(torques),
        "peak_velocity": max(abs(v) for v in velocities),
        "nonselected_drift": max(drift),
    }
    if not result["ok"]:
        result["failure"] = "settle_or_drift"
    return result


def evaluate(candidate: Candidate, seed: int) -> dict[str, Any]:
    scenarios = [
        (asdict(candidate), joint, amplitude, seed + joint * 100 + n)
        for joint in range(7)
        for n, amplitude in enumerate((-0.05, -0.03, -0.01, 0.01, 0.03, 0.05))
    ]
    trials = [_trial(item) for item in scenarios]
    valid = [x for x in trials if x.get("ok")]
    failure_rate = 1.0 - len(valid) / len(trials)
    p95 = float(np.percentile([x.get("p95_error", 1.0) for x in trials], 95))
    score = p95 + 0.05 * failure_rate + 0.02 * statistics.mean(
        x.get("overshoot", 1.0) for x in trials
    )
    return {
        "candidate": asdict(candidate),
        "score": score,
        "p95_error": p95,
        "failure_rate": failure_rate,
        "trials": trials,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def doctor(_: argparse.Namespace) -> int:
    report = {
        "model": str(root() / MODEL_REL),
        "model_exists": (root() / MODEL_REL).is_file(),
        "revision": REVISION,
        "cpu_count": os.cpu_count(),
        "gpu_used": False,
        "right_arm_joints": list(RIGHT_ARM_JOINT_NAMES),
        "physical_robot_commands": False,
    }
    try:
        model, _, _, _, _ = _load()
        report["mujoco_timestep"] = model.opt.timestep
        report["ok"] = True
    except Exception as exc:
        report["ok"] = False
        report["error"] = str(exc)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def smoke(args: argparse.Namespace) -> int:
    result = evaluate(Candidate(60.0, 1.5, 0.10, 0.50), args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    return 0 if result["failure_rate"] < 1.0 else 1


def _candidate(index: int, seed: int) -> Candidate:
    # Low-discrepancy-like deterministic sampling without materializing an unbounded grid.
    rng = random.Random(seed + index * 104729)
    return Candidate(
        kp=rng.uniform(30.0, 80.0),
        kd=rng.uniform(0.5, 4.0),
        vmax=rng.uniform(0.05, 0.25),
        amax=rng.uniform(0.25, 2.0),
    )


def sweep(args: argparse.Namespace) -> int:
    started = time.time()
    run = root() / OUTPUT_REL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True)
    latest = root() / OUTPUT_REL / "latest.json"
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_time = started
    index = 0
    deadline = started + args.max_hours * 3600
    while time.time() < deadline and (args.max_candidates is None or index < args.max_candidates):
        batch = [_candidate(index + i, args.seed) for i in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            evaluated = list(pool.map(evaluate, batch, range(args.seed + index, args.seed + index + len(batch))))
        for item in evaluated:
            results.append(item)
            if best is None or item["score"] < best["score"] * 0.99:
                best, best_time = item, time.time()
        index += len(batch)
        summary = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "elapsed_hours": (time.time() - started) / 3600,
            "evaluated": index,
            "best": best,
            "model_revision": REVISION,
            "real_robot_validated": False,
        }
        _atomic_json(run / "checkpoint.json", summary)
        _atomic_json(latest, {"run": str(run.relative_to(root()))})
        if time.time() - started >= args.min_hours * 3600 and time.time() - best_time >= args.plateau_hours * 3600:
            break
    summary["status"] = "complete"
    _atomic_json(run / "checkpoint.json", summary)
    compact = root() / "simulation/baselines/g1_right_arm"
    _atomic_json(compact / "best_candidate.json", best)
    _atomic_json(compact / "validation_summary.json", {k: v for k, v in summary.items() if k != "best"})
    print(json.dumps(summary, indent=2))
    return 0


def status(_: argparse.Namespace) -> int:
    latest = root() / OUTPUT_REL / "latest.json"
    if not latest.is_file():
        print("No sweep has been started.")
        return 1
    pointer = json.loads(latest.read_text())
    print((root() / pointer["run"] / "checkpoint.json").read_text(), end="")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.set_defaults(func=doctor)
    m = sub.add_parser("smoke")
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(func=smoke)
    w = sub.add_parser("sweep")
    w.add_argument("--workers", type=int, default=14)
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--min-hours", type=float, default=48.0)
    w.add_argument("--max-hours", type=float, default=72.0)
    w.add_argument("--plateau-hours", type=float, default=12.0)
    w.add_argument("--max-candidates", type=int)
    w.set_defaults(func=sweep)
    s = sub.add_parser("status")
    s.set_defaults(func=status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
