#!/usr/bin/env python3
"""Generate deterministic and held-out moving-bunny Isaac replay inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np


START_RIGHT_Q = (
    0.2891673744,
    -0.1298251152,
    0.0039188415,
    0.9780925512,
    -0.1113813892,
    -0.0022170816,
    -0.0082091941,
)
ORIGIN = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
AXIS_U = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
AXIS_V = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))
NORMAL = -np.cross(AXIS_U, AXIS_V)
NORMAL /= np.linalg.norm(NORMAL)
MINIMUM_UV = (0.0, -0.3545687169)
MAXIMUM_UV = (0.34, 0.3254312831)


def body_state() -> list[float]:
    q = [
        -0.05,
        0.0,
        0.0,
        0.2,
        -0.15,
        0.0,
        -0.05,
        0.0,
        0.0,
        0.2,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *START_RIGHT_Q,
    ]
    assert len(q) == 29
    return q


def support_plane(table_shift_x_m: float) -> dict[str, object]:
    origin = ORIGIN + np.asarray((table_shift_x_m, 0.0, 0.0))
    return {
        "normal": NORMAL.tolist(),
        "offset": -float(NORMAL @ origin),
        "footprint": {
            "origin": origin.tolist(),
            "axis_u": AXIS_U.tolist(),
            "axis_v": AXIS_V.tolist(),
            "minimum_uv": list(MINIMUM_UV),
            "maximum_uv": list(MAXIMUM_UV),
            "certified_edges": ["u_min"],
            "edge_sources": {"u_min": "calibrated_pixel_near_edge"},
            "lateral_margin_m": 0.07,
            "source": "captured_lab_matrix",
        },
    }


def episode(
    *,
    name: str,
    bunny_x_m: float,
    speed_m_s: float,
    table_shift_x_m: float,
    occlusion: bool,
    seed: int,
) -> dict[str, object]:
    if not 0.30 <= bunny_x_m <= 0.45:
        raise ValueError("bunny_x_m is outside the validated lane")
    if not 0.04 <= speed_m_s <= 0.08:
        raise ValueError("speed_m_s is outside the declared matrix")
    start_y_m = 0.30
    crossing_y_m = -0.10
    crossing_time_s = (start_y_m - crossing_y_m) / speed_m_s
    duration_s = crossing_time_s + 1.0
    frame_count = int(math.ceil(duration_s * 30.0)) + 1
    frames: list[dict[str, object]] = []
    for index in range(frame_count):
        time_s = min(duration_s, index / 30.0)
        hidden = occlusion and 2.0 <= time_s < 2.15
        frame: dict[str, object] = {
            "time_s": round(time_s, 6),
        }
        if index == 0:
            frame["measured_body_q_rad"] = body_state()
            frame["measured_body_dq_rad_s"] = [0.0] * 29
        if not hidden:
            frame.update(
                {
                    "track_id": 1,
                    "class_name": "bunny",
                    "confidence": 0.9,
                    "object_xyz_m": [
                        round(bunny_x_m, 6),
                        round(start_y_m - speed_m_s * time_s, 6),
                        0.15,
                    ],
                    "object_velocity_m_s": [0.0, -speed_m_s, 0.0],
                }
            )
        frames.append(frame)
    return {
        "schema_version": 1,
        "source": "generated_ballistic_intercept_matrix",
        "calibration_id": "lab-sim",
        "coordinate_frame": "g1_torso_x_forward_y_left_z_up",
        "right_side_direction": [0.0, -1.0, 0.0],
        "support_plane": support_plane(table_shift_x_m),
        "object_proxy": {
            "shape": "capsule",
            "radius_m": 0.055,
            # Isaac's capsule height excludes both hemispherical caps. Keep
            # the total proxy clear of the calibrated table at z=0.15.
            "height_m": 0.10,
        },
        "scenario": {
            "name": name,
            "seed": seed,
            "bunny_x_m": bunny_x_m,
            "speed_m_s": speed_m_s,
            "table_shift_x_m": table_shift_x_m,
            "occlusion": occlusion,
            "crossing_time_s": crossing_time_s,
        },
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--random-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--canonical-repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.random_count < 0 or args.canonical_repetitions < 1:
        parser.error("counts must be non-negative and repetitions must be positive")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for bunny_x_m in (0.33, 0.36, 0.40):
        for speed_m_s in (0.04, 0.05, 0.07):
            for table_shift_x_m in (0.0, -0.04):
                for occlusion in (False, True):
                    name = (
                        f"canonical_x{round(100 * bunny_x_m):02d}"
                        f"_v{round(100 * speed_m_s):02d}"
                        f"_table{'close' if table_shift_x_m < 0 else 'base'}"
                        f"_{'occ' if occlusion else 'clear'}"
                    )
                    value = episode(
                        name=name,
                        bunny_x_m=bunny_x_m,
                        speed_m_s=speed_m_s,
                        table_shift_x_m=table_shift_x_m,
                        occlusion=occlusion,
                        seed=args.seed,
                    )
                    path = output / f"{name}.json"
                    path.write_text(
                        json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    cases.append(
                        {
                            "name": name,
                            "path": path.name,
                            "kind": "canonical",
                            "repetitions": args.canonical_repetitions,
                            **value["scenario"],
                        }
                    )

    rng = random.Random(args.seed)
    for index in range(args.random_count):
        case_seed = rng.randrange(2**31)
        case_rng = random.Random(case_seed)
        name = f"heldout_{index:03d}_{case_seed}"
        value = episode(
            name=name,
            bunny_x_m=case_rng.uniform(0.31, 0.43),
            speed_m_s=case_rng.uniform(0.04, 0.075),
            table_shift_x_m=case_rng.uniform(-0.05, 0.02),
            occlusion=case_rng.random() < 0.35,
            seed=case_seed,
        )
        path = output / f"{name}.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cases.append(
            {
                "name": name,
                "path": path.name,
                "kind": "heldout",
                "repetitions": 1,
                **value["scenario"],
            }
        )

    manifest = {
        "schema_version": 1,
        "generator_seed": args.seed,
        "canonical_case_count": 36,
        "canonical_repetitions": args.canonical_repetitions,
        "heldout_case_count": args.random_count,
        "expected_episode_count": 36 * args.canonical_repetitions
        + args.random_count,
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
