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
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np

from object_tracking.arm_tracking.joints import RIGHT_ARM_JOINT_NAMES
from object_tracking.g1_exchange import exchange_candidates

REVISION = "ae6a8403e272733e9996ef59990880330496177f"
MODEL_REL = Path(".deps/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
OUTPUT_REL = Path("runs/simulation")
TORQUE_LIMITS = (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0)
JOINT_PRIORITY = (1.4, 1.4, 1.2, 1.6, 0.9, 0.9, 0.9)
SEARCH_BOUNDS = ((20.0, 110.0), (0.3, 6.0), (0.03, 0.35), (0.15, 4.0))
_CACHED_SIM: tuple[Any, list[int], list[int], list[int], np.ndarray, np.ndarray] | None = None


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


def _simulation() -> tuple[Any, Any, list[int], list[int], list[int]]:
    """Reuse one model per worker; forked immutable mesh pages stay shared."""
    import mujoco

    global _CACHED_SIM
    if _CACHED_SIM is None:
        model, _, qpos, qvel, actuators = _load()
        _CACHED_SIM = (
            model,
            qpos,
            qvel,
            actuators,
            model.opt.gravity.copy(),
            model.dof_damping.copy(),
        )
    model, qpos, qvel, actuators, gravity, damping = _CACHED_SIM
    model.opt.gravity[:] = gravity
    model.dof_damping[:] = damping
    return model, mujoco.MjData(model), qpos, qvel, actuators


def _trial(payload: tuple[dict[str, float], int, float, int]) -> dict[str, Any]:
    import mujoco

    raw, joint_index, amplitude, seed = payload
    c = Candidate(**raw)
    model, data, qpos, qvel, actuators = _simulation()
    rng = random.Random(seed)
    model.opt.gravity[2] *= rng.uniform(0.98, 1.02)
    model.dof_damping[:] *= rng.uniform(0.80, 1.20)
    mujoco.mj_forward(model, data)
    baseline = data.qpos[qpos].copy()
    requested = baseline.copy()
    requested[joint_index] += amplitude
    target = baseline.copy()
    velocity = np.zeros(7)
    delay_steps = rng.randint(0, 3)
    delayed_targets = [baseline.copy()] * delay_steps
    torque_scale = rng.uniform(0.88, 1.0)
    position_noise = rng.uniform(0.00025, 0.001)
    velocity_noise = rng.uniform(0.002, 0.01)
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
            delayed_targets.append(target.copy())
        control_target = delayed_targets.pop(0) if delayed_targets else target
        q = data.qpos[qpos] + np.asarray(
            [rng.uniform(-position_noise, position_noise) for _ in qpos]
        )
        dq = data.qvel[qvel] + np.asarray(
            [rng.uniform(-velocity_noise, velocity_noise) for _ in qvel]
        )
        tau = torque_scale * (c.kp * (control_target - q) - c.kd * dq)
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
        "domain_randomization": {
            "delay_steps": delay_steps,
            "torque_scale": torque_scale,
            "position_noise_rad": position_noise,
            "velocity_noise_rad_s": velocity_noise,
        },
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
    weighted_errors = [
        x.get("p95_error", 1.0)
        * JOINT_PRIORITY[RIGHT_ARM_JOINT_NAMES.index(x.get("joint", RIGHT_ARM_JOINT_NAMES[0]))]
        for x in trials
    ]
    p95 = float(np.percentile(weighted_errors, 95))
    shoulder_elbow_p95 = float(
        np.percentile(
            [
                x.get("p95_error", 1.0)
                for x in trials
                if x.get("joint") in RIGHT_ARM_JOINT_NAMES[:4]
            ],
            95,
        )
    )
    score = p95 + 0.05 * failure_rate + 0.02 * statistics.mean(
        x.get("overshoot", 1.0) for x in trials
    )
    return {
        "candidate": asdict(candidate),
        "score": score,
        "p95_error": p95,
        "shoulder_elbow_p95_error": shoulder_elbow_p95,
        "failure_rate": failure_rate,
        "trials": trials,
    }


def evaluate_repeated(payload: tuple[Candidate, int, int]) -> dict[str, Any]:
    """Score a candidate by its worst practical randomized replay metrics."""
    candidate, seed, replays = payload
    samples = [evaluate(candidate, seed + replay * 100_003) for replay in range(replays)]
    if replays == 1:
        return samples[0]
    scores = [sample["score"] for sample in samples]
    errors = [sample["p95_error"] for sample in samples]
    shoulder_elbow = [sample["shoulder_elbow_p95_error"] for sample in samples]
    failures = [sample["failure_rate"] for sample in samples]
    return {
        "candidate": asdict(candidate),
        "score": float(np.percentile(scores, 90)),
        "p95_error": float(np.percentile(errors, 90)),
        "shoulder_elbow_p95_error": float(np.percentile(shoulder_elbow, 90)),
        "failure_rate": float(max(failures)),
        "randomized_replays": replays,
        "trials": [trial for sample in samples for trial in sample["trials"]],
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


def _candidate(index: int, seed: int, elites: list[dict[str, float]] = []) -> Candidate:
    """Random candidate with optional local refinement around a shared elite."""
    rng = random.Random(seed + index * 104729)
    candidate = Candidate(
        kp=rng.uniform(*SEARCH_BOUNDS[0]),
        kd=rng.uniform(*SEARCH_BOUNDS[1]),
        vmax=rng.uniform(*SEARCH_BOUNDS[2]),
        amax=rng.uniform(*SEARCH_BOUNDS[3]),
    )
    if elites and index % 2:
        elite = elites[index % len(elites)]
        candidate = Candidate(
            kp=np.clip(rng.gauss(elite["kp"], 7.0), *SEARCH_BOUNDS[0]),
            kd=np.clip(rng.gauss(elite["kd"], 0.5), *SEARCH_BOUNDS[1]),
            vmax=np.clip(rng.gauss(elite["vmax"], 0.035), *SEARCH_BOUNDS[2]),
            amax=np.clip(rng.gauss(elite["amax"], 0.4), *SEARCH_BOUNDS[3]),
        )
    return candidate


def _candidate_batch(
    index: int,
    count: int,
    seed: int,
    *,
    method: str,
    elites: list[dict[str, float]],
    best: dict[str, Any] | None,
) -> list[Candidate]:
    if method == "random":
        return [_candidate(index + offset, seed, elites) for offset in range(count)]
    if method == "exploit":
        seeds = elites + ([] if best is None else [best["candidate"]])
        return [_candidate(index + offset, seed + 7919, seeds) for offset in range(count)]
    if method != "sobol":
        raise ValueError(f"unknown search method: {method}")
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
        if index:
            sampler.fast_forward(index)
        points = qmc.scale(sampler.random(count), *zip(*SEARCH_BOUNDS, strict=True))
    except ModuleNotFoundError:
        # Keep the minimal local environment functional; sim hosts install SciPy.
        primes = (2, 3, 5, 7)

        def radical_inverse(value: int, base: int) -> float:
            fraction = 0.0
            denominator = 1.0
            while value:
                denominator *= base
                value, remainder = divmod(value, base)
                fraction += remainder / denominator
            return fraction

        unit = np.asarray(
            [
                [radical_inverse(index + offset + seed + 1, base) for base in primes]
                for offset in range(count)
            ]
        )
        points = np.asarray(
            [
                lower + unit[:, dimension] * (upper - lower)
                for dimension, (lower, upper) in enumerate(SEARCH_BOUNDS)
            ]
        ).T
    return [Candidate(*map(float, point)) for point in points]


def sweep(args: argparse.Namespace) -> int:
    started = time.time()
    run = root() / OUTPUT_REL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True)
    latest = root() / OUTPUT_REL / "latest.json"
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_time = started
    index = 0
    iteration = 0
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if not methods or any(item not in {"sobol", "random", "exploit"} for item in methods):
        raise SystemExit("--methods must contain sobol, random, and/or exploit")
    method_best: dict[str, dict[str, Any]] = {}
    deadline = started + args.max_hours * 3600
    _simulation()
    process_context = multiprocessing.get_context("fork")
    while time.time() < deadline and (args.max_candidates is None or index < args.max_candidates):
        import psutil

        if psutil.virtual_memory().available < args.min_available_mib * 1024 * 1024:
            print(
                f"memory guard: waiting for {args.min_available_mib} MiB available RAM",
                flush=True,
            )
            time.sleep(5)
            continue
        elites = exchange_candidates(args.exchange_ref)
        method = methods[iteration % len(methods)]
        batch = _candidate_batch(
            index, args.workers, args.seed, method=method, elites=elites, best=best
        )
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=process_context
        ) as pool:
            payloads = [
                (candidate, args.seed + candidate_index, args.replays_per_candidate)
                for candidate_index, candidate in enumerate(batch, start=index)
            ]
            evaluated = list(pool.map(evaluate_repeated, payloads))
        for item in evaluated:
            results.append(item)
            item["method"] = method
            if method not in method_best or item["score"] < method_best[method]["score"]:
                method_best[method] = item
            if best is None or item["score"] < best["score"] * 0.99:
                best, best_time = item, time.time()
        index += len(batch)
        iteration += 1
        summary = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "elapsed_hours": (time.time() - started) / 3600,
            "evaluated": index,
            "best": best,
            "model_revision": REVISION,
            "real_robot_validated": False,
            "exchange_elites": len(elites),
            "method": method,
            "method_best": method_best,
            "minimum_hours": args.min_hours,
            "maximum_hours": args.max_hours,
            "plateau_hours": args.plateau_hours,
            "no_material_improvement_hours": (time.time() - best_time) / 3600,
            "replays_per_candidate": args.replays_per_candidate,
        }
        _atomic_json(run / "checkpoint.json", summary)
        _atomic_json(latest, {"run": str(run.relative_to(root()))})
        elapsed = max(time.time() - started, 0.001)
        print(
            f"[{summary['elapsed_hours']:.2f}h] evaluated={index} "
            f"rate={index / elapsed * 3600:.0f}/h "
            f"method={method} best_score={best['score']:.6f} failures={best['failure_rate']:.1%}",
            flush=True,
        )
        if time.time() - started >= args.min_hours * 3600 and time.time() - best_time >= args.plateau_hours * 3600:
            print(
                f"stopping: no material best-score improvement for "
                f"{args.plateau_hours:g}h after the {args.min_hours:g}h minimum",
                flush=True,
            )
            break
    summary["status"] = "complete"
    _atomic_json(run / "checkpoint.json", summary)
    compact = root() / "simulation/baselines/g1_right_arm"
    _atomic_json(compact / "best_candidate.json", best)
    _atomic_json(compact / "validation_summary.json", {k: v for k, v in summary.items() if k != "best"})
    print(json.dumps(summary, indent=2))
    return 0


def status(args: argparse.Namespace) -> int:
    latest = root() / OUTPUT_REL / "latest.json"
    if not latest.is_file():
        print("No sweep has been started.")
        return 1
    pointer = json.loads(latest.read_text())
    checkpoint = json.loads((root() / pointer["run"] / "checkpoint.json").read_text())
    if args.verbose:
        print(json.dumps(checkpoint, indent=2, sort_keys=True))
        return 0
    elapsed = float(checkpoint.get("elapsed_hours", 0.0))
    evaluated = int(checkpoint.get("evaluated", 0))
    best = checkpoint.get("best") or {}
    compact = {
        "status": checkpoint.get("status"),
        "started_at": checkpoint.get("started_at"),
        "elapsed_hours": round(elapsed, 3),
        "minimum_hours": checkpoint.get("minimum_hours"),
        "maximum_hours": checkpoint.get("maximum_hours"),
        "minimum_remaining_hours": (
            round(max(0.0, float(checkpoint["minimum_hours"]) - elapsed), 3)
            if checkpoint.get("minimum_hours") is not None
            else None
        ),
        "plateau_hours": checkpoint.get("plateau_hours"),
        "no_material_improvement_hours": round(
            float(checkpoint.get("no_material_improvement_hours", 0.0)), 3
        ),
        "evaluated": evaluated,
        "candidates_per_hour": round(evaluated / elapsed) if elapsed else 0,
        "best_candidate": best.get("candidate"),
        "best_score": best.get("score"),
        "best_p95_error": best.get("p95_error"),
        "best_failure_rate": best.get("failure_rate"),
        "real_robot_validated": checkpoint.get("real_robot_validated", False),
    }
    print(json.dumps(compact, indent=2))
    return 0


def validate(args: argparse.Namespace) -> int:
    """Stress-test the latest simulated candidate under independent random seeds.

    This is deliberately a read-only simulation gate, not a gain promotion path.
    A single favorable randomized rollout is not evidence that a candidate will
    transfer to the physical G1.
    """
    latest = root() / OUTPUT_REL / "latest.json"
    if not latest.is_file():
        raise SystemExit("No sweep checkpoint exists to validate.")
    pointer = json.loads(latest.read_text())
    checkpoint = json.loads((root() / pointer["run"] / "checkpoint.json").read_text())
    best = checkpoint.get("best") or {}
    raw = best.get("candidate")
    if not isinstance(raw, dict):
        raise SystemExit("Latest sweep has no candidate to validate.")
    candidate = Candidate(**raw)
    _simulation()
    context = multiprocessing.get_context("fork")
    seeds = [args.seed + index * 100_003 for index in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        samples = list(pool.map(evaluate, [candidate] * len(seeds), seeds))
    scores = [sample["score"] for sample in samples]
    errors = [sample["p95_error"] for sample in samples]
    failures = [sample["failure_rate"] for sample in samples]
    report = {
        "schema_version": 1,
        "source_run": pointer["run"],
        "candidate": raw,
        "randomized_replays": len(samples),
        "score_median": float(statistics.median(scores)),
        "score_p95": float(np.percentile(scores, 95)),
        "p95_error_median": float(statistics.median(errors)),
        "p95_error_p95": float(np.percentile(errors, 95)),
        "failure_rate_mean": float(statistics.mean(failures)),
        "failure_rate_worst": float(max(failures)),
        "real_robot_validated": False,
        "promotion": "blocked_pending_guarded_encoder_telemetry",
    }
    destination = root() / pointer["run"] / "robust_validation.json"
    _atomic_json(destination, report)
    print(json.dumps(report, indent=2))
    return 0


def calibrate(args: argparse.Namespace) -> int:
    """Summarize guarded real encoder telemetry; never contacts the robot."""
    source = Path(args.telemetry)
    if source.suffix == ".jsonl":
        raw_samples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        samples = []
        for sample in raw_samples:
            measured = sample.get("measured_arm_q")
            commanded = sample.get("commanded_arm_q")
            if not isinstance(measured, list) or not isinstance(commanded, list):
                continue
            if len(measured) == 14 and len(commanded) == 14:
                measured, commanded = measured[-7:], commanded[-7:]
            timestamp = sample.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            samples.append(
                {"timestamp": timestamp, "measured_q": measured, "commanded_q": commanded}
            )
    else:
        payload = json.loads(source.read_text())
        samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not samples:
        raise SystemExit("telemetry must contain non-empty commanded and measured joint samples")
    timestamps = [float(sample["timestamp"]) for sample in samples]
    measured = [sample["measured_q"] for sample in samples]
    commanded = [sample["commanded_q"] for sample in samples]
    if any(len(values) != 7 for values in measured + commanded):
        raise SystemExit("every measured_q and commanded_q sample must contain seven right-arm joints")
    errors = np.asarray(commanded, dtype=float) - np.asarray(measured, dtype=float)
    summary = {
        "schema_version": 1,
        "source": str(source),
        "samples": len(samples),
        "duration_s": timestamps[-1] - timestamps[0],
        "sample_hz": (len(samples) - 1) / max(timestamps[-1] - timestamps[0], 1e-9),
        "per_joint_rmse_rad": np.sqrt(np.mean(errors * errors, axis=0)).tolist(),
        "per_joint_p95_error_rad": np.percentile(np.abs(errors), 95, axis=0).tolist(),
        "real_robot_validated": False,
        "next_step": "review this telemetry before promoting any simulated gain",
    }
    destination = root() / "runs/simulation/telemetry/latest_fit.json"
    _atomic_json(destination, summary)
    print(json.dumps(summary, indent=2))
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
    w.add_argument(
        "--replays-per-candidate",
        type=int,
        default=1,
        help="independent randomized replays aggregated into every candidate score",
    )
    w.add_argument(
        "--min-available-mib",
        type=int,
        default=2048,
        help="pause before launching a batch when available system RAM is lower",
    )
    w.add_argument(
        "--methods",
        default="sobol,random,exploit",
        help="comma-separated global/local search methods cycled per batch",
    )
    w.add_argument(
        "--exchange-ref",
        default="origin/sim:simulation/exchange/elite_candidates.json",
        help="Git object containing compact cross-host candidate elites",
    )
    w.set_defaults(func=sweep)
    s = sub.add_parser("status")
    s.add_argument("--verbose", action="store_true", help="include every trial in the checkpoint")
    s.set_defaults(func=status)
    v = sub.add_parser("validate")
    v.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    v.add_argument("--seeds", type=int, default=64)
    v.add_argument("--seed", type=int, default=10_000)
    v.set_defaults(func=validate)
    c = sub.add_parser("calibrate")
    c.add_argument("telemetry", help="guarded seven-joint encoder telemetry JSON")
    c.set_defaults(func=calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")
    if getattr(args, "replays_per_candidate", 1) < 1:
        raise SystemExit("--replays-per-candidate must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
