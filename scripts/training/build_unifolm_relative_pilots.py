#!/usr/bin/env python3
"""Build matched absolute/relative UniFoLM pilots for the right-hand task."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from object_tracking.unifolm_relative_actions import RELATIVE_POSE23_V1


RIGHT_POSE = range(9, 18)


def right_only_loss_weights() -> list[float]:
    return [1.0 if index in RIGHT_POSE else 0.0 for index in range(23)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_variants(
    base: dict[str, Any],
    *,
    relative_statistics: Path,
    max_train_steps: int,
) -> dict[str, dict[str, Any]]:
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    variants = {}
    for representation in ("absolute_pose23", RELATIVE_POSE23_V1):
        prefix = "absolute" if representation == "absolute_pose23" else "relative"
        for temporal, window_size, stride in (
            ("t1", 1, 1),
            ("t5s3", 5, 3),
        ):
            name = f"{prefix}-{temporal}-right9"
            config = deepcopy(base)
            data = config["datasets"]["vla_data"]
            data["window_size"] = window_size
            data["observation_stride"] = stride
            data["action_representation"] = representation
            if representation == RELATIVE_POSE23_V1:
                data["relative_action_statistics"] = str(
                    relative_statistics.resolve()
                )
            else:
                data.pop("relative_action_statistics", None)
            config["framework"]["action_model"][
                "loss_dimension_weights"
            ] = right_only_loss_weights()
            config["framework"]["action_model"]["use_relative_action"] = (
                representation == RELATIVE_POSE23_V1
            )
            trainer = config["trainer"]
            trainer["max_train_steps"] = max_train_steps
            trainer["num_warmup_steps"] = min(100, max(10, max_train_steps // 20))
            trainer["save_interval"] = max_train_steps
            trainer["eval_interval"] = max_train_steps
            trainer["learning_rate"]["base"] = 3e-5
            trainer["learning_rate"]["action_model"] = 3e-5
            trainer["is_resume"] = False
            trainer["resume_epoch"] = None
            trainer["resume_step"] = None
            config["run_id"] = f"pose23-pilot-{name}"
            variants[name] = config
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--relative-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=750)
    args = parser.parse_args()

    statistics = json.loads(args.relative_statistics.read_text(encoding="utf-8"))
    if statistics.get("representation", {}).get("version") != RELATIVE_POSE23_V1:
        raise ValueError("relative statistics use the wrong representation")
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    variants = build_variants(
        base,
        relative_statistics=args.relative_statistics,
        max_train_steps=args.max_train_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "physical_robot_authorized": False,
        "relative_statistics_sha256": sha256(args.relative_statistics),
        "variants": {},
    }
    for name, config in variants.items():
        path = args.output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest["variants"][name] = {
            "config": str(path.resolve()),
            "representation": config["datasets"]["vla_data"][
                "action_representation"
            ],
            "window_size": config["datasets"]["vla_data"]["window_size"],
            "observation_stride": config["datasets"]["vla_data"][
                "observation_stride"
            ],
            "max_train_steps": args.max_train_steps,
        }
    manifest_path = args.output_dir / "PILOT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"UNIFOLM_RELATIVE_PILOTS_READY output={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
