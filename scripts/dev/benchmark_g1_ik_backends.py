#!/usr/bin/env python3
"""Offline benchmark for G1 IK candidate generators.

The canonical pinned XR-Teleoperate URDF and this project's FK/collision model
judge every result. The optional YuehChuan backend is imported from a separate
checkout because that repository does not publish a redistribution license.
This script never opens ROS, DDS, cameras, or robot transports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records]
    return {
        "samples": len(records),
        "accepted": sum(bool(record["accepted"]) for record in records),
        "within_5mm_3deg": sum(bool(record["within_5mm_3deg"]) for record in records),
        "unsafe_outputs": sum(bool(record["unsafe"]) for record in records),
        "position_error_m_mean": float(
            np.mean([record["position_error_m"] for record in records])
        ),
        "position_error_m_p95": _percentile(
            [float(record["position_error_m"]) for record in records], 95
        ),
        "orientation_error_rad_mean": float(
            np.mean([record["orientation_error_rad"] for record in records])
        ),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
    }


def _pose_errors(
    solver: G1RightArmIK,
    target: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float]:
    actual = solver.forward_kinematics(q)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    orientation = float(
        np.linalg.norm(solver.pin.log3(actual[:3, :3] @ target[:3, :3].T))
    )
    return position, orientation


def _record(
    solver: G1RightArmIK,
    target: np.ndarray,
    q: np.ndarray | None,
    *,
    accepted: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    if q is None or q.shape != (7,) or not np.all(np.isfinite(q)):
        return {
            "accepted": False,
            "within_5mm_3deg": False,
            "unsafe": True,
            "position_error_m": float("inf"),
            "orientation_error_rad": float("inf"),
            "latency_ms": elapsed_s * 1000.0,
        }
    position, orientation = _pose_errors(solver, target, q)
    unsafe = bool(
        np.any(q < solver.lower)
        or np.any(q > solver.upper)
        or solver._collision_pairs(q)
    )
    return {
        "accepted": accepted,
        "within_5mm_3deg": position <= 0.005 and orientation <= np.deg2rad(3.0),
        "unsafe": unsafe,
        "position_error_m": position,
        "orientation_error_rad": orientation,
        "latency_ms": elapsed_s * 1000.0,
    }


def _load_yueh_solver(checkout: Path) -> Any:
    source = checkout / "g1_ik_solver.py"
    xml = checkout / "models" / "g1_description" / "g1_dual_arm.xml"
    if not source.is_file() or not xml.is_file():
        raise FileNotFoundError("YuehChuan checkout is missing solver or model files")
    spec = importlib.util.spec_from_file_location("yueh_unitree_g1_ik", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1IKSolver(str(xml))


def _benchmark_global(
    canonical: G1RightArmIK,
    targets: list[np.ndarray],
    seed_q: np.ndarray,
    yueh: Any | None,
) -> dict[str, Any]:
    backends: dict[str, Callable[[np.ndarray], tuple[np.ndarray | None, bool]]] = {}

    def canonical_solve(target: np.ndarray) -> tuple[np.ndarray | None, bool]:
        result = canonical.solve(target, seed_q)
        return (
            None if result.q_rad is None else np.asarray(result.q_rad, dtype=float),
            result.ok,
        )

    backends[f"canonical_{canonical.global_backend}"] = canonical_solve
    if yueh is not None:
        right_qadr = np.asarray(
            [
                yueh.model.jnt_qposadr[yueh.model.joint(name).id]
                for name in (
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                )
            ],
            dtype=int,
        )
        full_seed = yueh.qpos_home.copy()
        full_seed[right_qadr] = seed_q

        def yueh_solve(target: np.ndarray) -> tuple[np.ndarray | None, bool]:
            quaternion = Rotation.from_matrix(target[:3, :3]).as_quat()
            full_q = yueh.solve(
                "right",
                target[:3, 3],
                quaternion,
                full_seed,
                max_iters=100,
                err_thresh=1e-4,
            )
            # The upstream API has no convergence status, so "accepted" means
            # only that it returned a finite vector. Canonical metrics below
            # determine whether that output was accurate and safe.
            q = np.asarray(full_q[right_qadr], dtype=float)
            return q, bool(np.all(np.isfinite(q)))

        backends["yueh_mujoco_dls_reference"] = yueh_solve

    output: dict[str, Any] = {}
    for name, solve in backends.items():
        records: list[dict[str, Any]] = []
        for target in targets:
            started = time.perf_counter()
            try:
                q, accepted = solve(target)
            except Exception:
                q, accepted = None, False
            records.append(
                _record(
                    canonical,
                    target,
                    q,
                    accepted=accepted,
                    elapsed_s=time.perf_counter() - started,
                )
            )
        output[name] = _summary(records)
    return output


def _benchmark_local_servo(
    canonical: G1RightArmIK,
    targets: list[np.ndarray],
    seed_q: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    initial_position = canonical.forward_kinematics(seed_q)[:3, 3]
    for backend in ("analytic", "finite_difference"):
        records: list[dict[str, Any]] = []
        progress: list[float] = []
        for target in targets:
            initial_error = float(np.linalg.norm(target[:3, 3] - initial_position))
            started = time.perf_counter()
            result = canonical.solve_local_translation(
                target,
                seed_q,
                validate_path=False,
                jacobian_backend=backend,
            )
            elapsed = time.perf_counter() - started
            q = None if result.q_rad is None else np.asarray(result.q_rad, dtype=float)
            record = _record(
                canonical,
                target,
                q,
                accepted=result.ok,
                elapsed_s=elapsed,
            )
            records.append(record)
            progress.append(
                0.0
                if not np.isfinite(record["position_error_m"]) or initial_error <= 0.0
                else 1.0 - float(record["position_error_m"]) / initial_error
            )
        summary = _summary(records)
        summary["mean_error_reduction_fraction"] = float(np.mean(progress))
        output[backend] = summary
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--yueh-checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    canonical = G1RightArmIK(default_urdf_path(args.repo_root.resolve()))
    seed_q = np.zeros(7, dtype=float)
    rng = np.random.default_rng(args.seed)
    global_targets: list[np.ndarray] = []
    local_targets: list[np.ndarray] = []
    seed_transform = canonical.forward_kinematics(seed_q)
    for _ in range(args.samples):
        goal_q = np.clip(
            seed_q + rng.normal(0.0, 0.055, 7),
            canonical.lower,
            canonical.upper,
        )
        global_targets.append(canonical.forward_kinematics(goal_q))
        target = seed_transform.copy()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        target[:3, 3] += direction * rng.uniform(0.002, 0.012)
        local_targets.append(target)

    yueh = None
    if args.yueh_checkout is not None:
        yueh = _load_yueh_solver(args.yueh_checkout.resolve())
    report = {
        "schema_version": 1,
        "robot_execution_authorized": False,
        "canonical_urdf": str(default_urdf_path(args.repo_root.resolve())),
        "canonical_global_backend": canonical.global_backend,
        "canonical_local_backend": canonical.local_backend,
        "samples": args.samples,
        "global_pose": _benchmark_global(canonical, global_targets, seed_q, yueh),
        "local_translation_step": _benchmark_local_servo(
            canonical,
            local_targets,
            seed_q,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
