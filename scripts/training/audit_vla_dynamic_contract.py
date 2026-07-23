#!/usr/bin/env python3
"""Audit temporal, action, contact, and normalization contracts for the VLA.

This is intentionally offline.  It reads immutable canonical HDF5 episodes and
never imports ROS, Unitree SDK, or Isaac runtime modules.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RIGHT_XYZ_17 = slice(6, 9)
LEFT_POSE_17 = slice(0, 6)
WAIST_AND_GRIPPERS_17 = slice(12, 17)
RIGHT_XYZ_23 = slice(9, 12)


def pose17_to_pose23(values: np.ndarray) -> np.ndarray:
    """Convert xyz+rpy poses to the Unitree two-column rotation representation."""

    poses = np.asarray(values, dtype=np.float32)
    if poses.shape[-1] != 17:
        raise ValueError(f"expected pose17, received {poses.shape}")

    def rotation6d(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        first = np.stack((cy * cp, sy * cp, -sp), axis=-1)
        second = np.stack(
            (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
            axis=-1,
        )
        return np.concatenate((first, second), axis=-1).astype(np.float32)

    left = np.concatenate((poses[..., 0:3], rotation6d(poses[..., 3:6])), axis=-1)
    right = np.concatenate((poses[..., 6:9], rotation6d(poses[..., 9:12])), axis=-1)
    return np.concatenate((left, right, poses[..., 12:17]), axis=-1).astype(np.float32)


def padded_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Return [T,H,D] causal targets with terminal absolute-action padding."""

    values = np.asarray(actions)
    if values.ndim != 2 or len(values) == 0 or horizon < 1:
        raise ValueError("actions must be non-empty [T,D] and horizon must be positive")
    indices = np.minimum(
        np.arange(len(values))[:, None] + np.arange(horizon)[None, :],
        len(values) - 1,
    )
    return values[indices]


def lag_errors(
    actions_xyz: np.ndarray,
    states_xyz: np.ndarray,
    lags: Iterable[int],
) -> dict[int, np.ndarray]:
    """Measure action[t] against observed state[t+lag] for each valid lag."""

    action = np.asarray(actions_xyz, dtype=np.float64)
    state = np.asarray(states_xyz, dtype=np.float64)
    if action.shape != state.shape or action.ndim != 2 or action.shape[1] != 3:
        raise ValueError("action and state positions must be matching [T,3] arrays")
    output: dict[int, np.ndarray] = {}
    for lag in lags:
        if lag >= 0:
            action_part = action[: len(action) - lag or None]
            state_part = state[lag:]
        else:
            action_part = action[-lag:]
            state_part = state[: len(state) + lag]
        output[int(lag)] = np.linalg.norm(action_part - state_part, axis=1)
    return output


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _resolve_episode(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # Manifest paths are workspace-relative, while root is the canonical dir.
    for base in (root.parents[1], root.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return root.parents[1] / candidate


def main() -> int:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "h5py is required to audit canonical episode files"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.05)
    parser.add_argument(
        "--enforce-demonstration-clearance",
        action="store_true",
        help=(
            "fail episodes below --minimum-clearance-m; disabled by default because "
            "deployment clearance is enforced by the geometric IK gateway"
        ),
    )
    parser.add_argument("--minimum-fps", type=float, default=28.0)
    parser.add_argument("--maximum-fps", type=float, default=32.0)
    parser.add_argument("--max-episodes-per-source", type=int)
    args = parser.parse_args()

    manifest_path = args.canonical_root / "CANONICAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    statistics = json.loads(args.stats.read_text())
    q01_action = np.asarray(statistics["action"]["q01"], dtype=np.float32)
    q99_action = np.asarray(statistics["action"]["q99"], dtype=np.float32)
    q01_state = np.asarray(statistics["proprio"]["q01"], dtype=np.float32)
    q99_state = np.asarray(statistics["proprio"]["q99"], dtype=np.float32)
    if any(array.shape != (23,) for array in (q01_action, q99_action, q01_state, q99_state)):
        raise ValueError("shared statistics must contain 23-D action and proprio bounds")

    episode_records: list[dict[str, Any]] = []
    lag_values: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    split_targets: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    split_states: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    counts_by_source: dict[str, int] = defaultdict(int)
    contact_mismatches = 0
    inactive_dimension_mismatches = 0
    clearance_failures = 0
    timestamp_failures = 0
    contactless_sim_episodes = 0

    for entry in manifest["episodes"]:
        source = str(entry["source"])
        if (
            args.max_episodes_per_source is not None
            and counts_by_source[source] >= args.max_episodes_per_source
        ):
            continue
        counts_by_source[source] += 1
        path = _resolve_episode(args.canonical_root, entry["path"])
        with h5py.File(path, "r") as episode:
            timestamps = np.asarray(episode["timestamp"][:], dtype=np.float64)
            states17 = np.asarray(episode["observations/ee_qpos"][:], dtype=np.float32)
            actions17 = np.asarray(episode["ee_action"][:], dtype=np.float32)
            contacts = np.asarray(
                episode["annotations/right_palm_contact"][:], dtype=bool
            )
            if (
                states17.shape != actions17.shape
                or states17.ndim != 2
                or states17.shape[1] != 17
                or len(timestamps) != len(states17)
                or len(contacts) != len(states17)
            ):
                raise ValueError(f"{path}: inconsistent canonical shapes")
            dt = np.diff(timestamps)
            fps = float(1.0 / np.median(dt))
            timestamp_ok = bool(
                np.all(dt > 0) and args.minimum_fps <= fps <= args.maximum_fps
            )
            timestamp_failures += int(not timestamp_ok)

            left_error = float(
                np.max(np.abs(actions17[:, LEFT_POSE_17] - states17[:, LEFT_POSE_17]))
            )
            held_error = float(
                np.max(
                    np.abs(
                        actions17[:, WAIST_AND_GRIPPERS_17]
                        - states17[:, WAIST_AND_GRIPPERS_17]
                    )
                )
            )
            if max(left_error, held_error) > 1e-5:
                inactive_dimension_mismatches += 1

            source_key = "sim" if source == "isaac" else "real"
            for lag, errors in lag_errors(
                actions17[:, RIGHT_XYZ_17],
                states17[:, RIGHT_XYZ_17],
                range(-5, 16),
            ).items():
                lag_values[source_key][lag].extend(errors.tolist())

            states23 = pose17_to_pose23(states17)
            actions23 = pose17_to_pose23(actions17)
            chunks = padded_action_chunks(actions23, args.horizon)
            split = str(entry["split"]).replace("validation", "val")
            split_targets[(source_key, split)].append(chunks[:, :, RIGHT_XYZ_23])
            split_states[(source_key, split)].append(states23[:, RIGHT_XYZ_23])

            action_outside = np.logical_or(
                actions23 < q01_action[None, :],
                actions23 > q99_action[None, :],
            )
            state_outside = np.logical_or(
                states23 < q01_state[None, :],
                states23 > q99_state[None, :],
            )

            minimum_clearance = episode.attrs.get("min_enabled_palm_clearance_m")
            if source_key == "sim":
                if not contacts.any():
                    contactless_sim_episodes += 1
                expected = (
                    np.linalg.norm(
                        np.asarray(
                            episode["sim_signals/contact_force"][:, :3],
                            dtype=np.float32,
                        ),
                        axis=1,
                    )
                    > float(episode.attrs.get("contact_force_threshold_n", 0.5))
                )
                contact_mismatches += int(np.count_nonzero(expected != contacts))
                if (
                    minimum_clearance is None
                    or float(minimum_clearance) < args.minimum_clearance_m
                ):
                    clearance_failures += 1

            episode_records.append(
                {
                    "path": str(path),
                    "source": source_key,
                    "split": split,
                    "frames": len(states17),
                    "fps": fps,
                    "timestamp_jitter_p95_ms": float(
                        np.percentile(np.abs(dt - np.median(dt)), 95) * 1000
                    ),
                    "left_held_max_error": left_error,
                    "waist_gripper_held_max_error": held_error,
                    "contact_frames": int(np.count_nonzero(contacts)),
                    "minimum_enabled_palm_clearance_m": (
                        None if minimum_clearance is None else float(minimum_clearance)
                    ),
                    "action_outside_shared_q01_q99_fraction": float(
                        np.mean(action_outside)
                    ),
                    "state_outside_shared_q01_q99_fraction": float(
                        np.mean(state_outside)
                    ),
                }
            )

    alignment: dict[str, Any] = {}
    for source, values_by_lag in lag_values.items():
        summaries = {
            str(lag): _percentiles(values)
            for lag, values in sorted(values_by_lag.items())
        }
        best_lag = min(
            values_by_lag,
            key=lambda lag: float(np.median(values_by_lag[lag])),
        )
        alignment[source] = {
            "best_lag_frames": int(best_lag),
            "best_lag_median_error_m": float(
                np.median(values_by_lag[best_lag])
            ),
            "by_lag": summaries,
        }

    baselines: dict[str, Any] = {}
    for key, target_parts in sorted(split_targets.items()):
        source, split = key
        targets = np.concatenate(target_parts, axis=0)
        states = np.concatenate(split_states[key], axis=0)
        current_prediction = np.repeat(states[:, None, :], args.horizon, axis=1)
        current_error = np.linalg.norm(current_prediction - targets, axis=-1)
        baselines[f"{source}/{split}"] = {
            "samples": len(targets),
            "current_pose_right_xyz_ade_m": float(np.mean(current_error)),
            "current_pose_right_xyz_fde_m": float(np.mean(current_error[:, -1])),
        }

    aggregate = {
        "episodes": len(episode_records),
        "sources": dict(counts_by_source),
        "contact_mismatched_frames": contact_mismatches,
        "contactless_sim_episodes": contactless_sim_episodes,
        "clearance_failure_episodes": clearance_failures,
        "inactive_dimension_mismatch_episodes": inactive_dimension_mismatches,
        "timestamp_failure_episodes": timestamp_failures,
        "fps": _percentiles([record["fps"] for record in episode_records]),
        "timestamp_jitter_p95_ms": _percentiles(
            [record["timestamp_jitter_p95_ms"] for record in episode_records]
        ),
        "action_outside_shared_q01_q99_fraction": float(
            np.mean(
                [
                    record["action_outside_shared_q01_q99_fraction"]
                    for record in episode_records
                ]
            )
        ),
        "state_outside_shared_q01_q99_fraction": float(
            np.mean(
                [
                    record["state_outside_shared_q01_q99_fraction"]
                    for record in episode_records
                ]
            )
        ),
    }
    failures = []
    for condition, label in (
        (contact_mismatches, "synthetic contact labels disagree with physics"),
        (contactless_sim_episodes, "synthetic episodes without contact"),
        (
            clearance_failures if args.enforce_demonstration_clearance else 0,
            "synthetic episodes below the optional demonstration-clearance gate",
        ),
        (inactive_dimension_mismatches, "inactive action dimensions are not held"),
        (timestamp_failures, "episodes violate the timestamp/FPS contract"),
    ):
        if condition:
            failures.append(f"{label}: {condition}")
    for source, summary in alignment.items():
        if summary["best_lag_frames"] < 0:
            failures.append(
                f"{source} action/state alignment is acausal: "
                f"{summary['best_lag_frames']} frames"
            )

    report = {
        "schema_version": 1,
        "canonical_root": str(args.canonical_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "stats": str(args.stats.resolve()),
        "horizon": args.horizon,
        "aggregate": aggregate,
        "alignment": alignment,
        "baselines": baselines,
        "gates": {
            "minimum_clearance_m": args.minimum_clearance_m,
            "demonstration_clearance_enforced": args.enforce_demonstration_clearance,
            "fps_range": [args.minimum_fps, args.maximum_fps],
            "passed": not failures,
            "failures": failures,
            "physical_robot_authorized": False,
        },
        "episodes": episode_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        "DYNAMIC_CONTRACT_PASS" if not failures else "DYNAMIC_CONTRACT_FAIL",
        f"episodes={len(episode_records)}",
        f"failures={len(failures)}",
        f"output={args.output}",
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
