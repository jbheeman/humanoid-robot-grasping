#!/usr/bin/env python3
"""Build the real-prioritized UniFoLM config for moving-plush blocking."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml


RELATIVE_POSE23 = "anchored_relative_pose23_v1"
FUTURE_STATE = "achieved_future_state_v1"


def build_config(
    base: dict,
    *,
    data_root: Path,
    statistics: Path,
    max_train_steps: int,
) -> dict:
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    stats = json.loads(statistics.read_text(encoding="utf-8"))
    representation = stats.get("representation", {})
    alignment = representation.get("target_alignment", {})
    if representation.get("version") != RELATIVE_POSE23:
        raise ValueError("statistics do not use anchored relative Pose23")
    if alignment.get("version") != FUTURE_STATE or alignment.get("lookahead_frames") != 1:
        raise ValueError("statistics do not use one-frame achieved-state alignment")

    config = deepcopy(base)
    data = config["datasets"]["vla_data"]
    data.update(
        {
            "data_root_dir": str(data_root.resolve()),
            "data_mix": "g1_plush_touch_mixed",
            "window_size": 5,
            "observation_stride": 3,
            "image_aug": True,
            "action_representation": RELATIVE_POSE23,
            "relative_action_statistics": str(statistics.resolve()),
            "target_alignment": {
                "version": FUTURE_STATE,
                "lookahead_frames": 1,
                "terminal_policy": "drop_unobservable_targets",
            },
        }
    )
    action_model = config["framework"]["action_model"]
    action_model["use_relative_action"] = True
    action_model["loss_dimension_weights"] = [
        1.0 if 9 <= index < 18 else 0.0 for index in range(23)
    ]
    trainer = config["trainer"]
    trainer.update(
        {
            "max_train_steps": max_train_steps,
            "num_warmup_steps": min(200, max(10, max_train_steps // 20)),
            "save_interval": min(500, max_train_steps),
            "eval_interval": min(500, max_train_steps),
            "is_resume": False,
            "resume_epoch": None,
            "resume_step": None,
        }
    )
    trainer["learning_rate"]["base"] = 2e-5
    trainer["learning_rate"]["action_model"] = 2e-5
    config["run_id"] = "v29-67real-motion-history"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=4000)
    args = parser.parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = build_config(
        base,
        data_root=args.data_root,
        statistics=args.statistics,
        max_train_steps=args.max_train_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(args.output)
    print(f"UNIFOLM_MOTION_CONFIG_READY output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
