#!/usr/bin/env python3
"""Build small, gated UniFoLM pilots for moving-object temporal ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml


LEFT_AND_AUX_WEIGHT = 0.05
RIGHT_POSE_SLICE = slice(9, 18)


def right_pose_loss_weights() -> list[float]:
    weights = [LEFT_AND_AUX_WEIGHT] * 23
    weights[RIGHT_POSE_SLICE] = [1.0] * 9
    return weights


def build_variants(base: dict[str, Any], *, max_train_steps: int) -> dict[str, dict]:
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    specifications = {
        "t1-all23": {
            "window_size": 1,
            "observation_stride": 1,
            "weights": None,
        },
        "t1-right9": {
            "window_size": 1,
            "observation_stride": 1,
            "weights": right_pose_loss_weights(),
        },
        "t5s3-right9": {
            # Five 30 Hz observations at [-12,-9,-6,-3,0] span 0.4 seconds.
            "window_size": 5,
            "observation_stride": 3,
            "weights": right_pose_loss_weights(),
        },
    }
    variants = {}
    for name, specification in specifications.items():
        cfg = deepcopy(base)
        cfg["datasets"]["vla_data"]["window_size"] = specification["window_size"]
        cfg["datasets"]["vla_data"]["observation_stride"] = specification[
            "observation_stride"
        ]
        action_model = cfg["framework"]["action_model"]
        if specification["weights"] is None:
            action_model.pop("loss_dimension_weights", None)
        else:
            action_model["loss_dimension_weights"] = specification["weights"]
        trainer = cfg["trainer"]
        trainer["max_train_steps"] = max_train_steps
        trainer["num_warmup_steps"] = min(100, max(10, max_train_steps // 20))
        trainer["save_interval"] = min(250, max_train_steps)
        trainer["eval_interval"] = min(250, max_train_steps)
        trainer["learning_rate"]["base"] = 3e-5
        trainer["learning_rate"]["action_model"] = 3e-5
        trainer["is_resume"] = False
        trainer["resume_epoch"] = None
        trainer["resume_step"] = None
        cfg["run_id"] = f"dynamic-pilot-{name}"
        variants[name] = cfg
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=750)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    variants = build_variants(base, max_train_steps=args.max_train_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "base_config": str(args.base_config.resolve()),
        "selection_split": "validation",
        "test_split_sealed": True,
        "physical_robot_authorized": False,
        "variants": {},
    }
    for name, config in variants.items():
        path = args.output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest["variants"][name] = {
            "config": str(path.resolve()),
            "window_size": config["datasets"]["vla_data"]["window_size"],
            "observation_stride": config["datasets"]["vla_data"][
                "observation_stride"
            ],
            "history_span_s_at_30hz": (
                (config["datasets"]["vla_data"]["window_size"] - 1)
                * config["datasets"]["vla_data"]["observation_stride"]
                / 30.0
            ),
            "loss_dimension_weights": config["framework"]["action_model"].get(
                "loss_dimension_weights"
            ),
            "max_train_steps": args.max_train_steps,
        }
    manifest_path = args.output_dir / "PILOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"UNIFOLM_DYNAMIC_PILOTS_READY output={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
