#!/usr/bin/env python3
"""Build controlled v30 active-motion/per-horizon UniFoLM pilot configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml


PILOTS = {
    "mixed_active": {
        "data_mix": "g1_plush_touch_mixed",
        "steps": 6000,
        "learning_rate": 3e-5,
        "warmup": 300,
        "active_threshold": 0.01,
        "inactive_keep": 0.15,
    },
    "real_only_finish": {
        "data_mix": "g1_plush_touch_real",
        "steps": 3000,
        "learning_rate": 1e-5,
        "warmup": 150,
        "active_threshold": 0.01,
        "inactive_keep": 0.20,
    },
    "mixed_fine_motion": {
        "data_mix": "g1_plush_touch_mixed",
        "steps": 5000,
        "learning_rate": 2e-5,
        "warmup": 250,
        "active_threshold": 0.005,
        "inactive_keep": 0.25,
    },
}


def build(base: dict, statistics: Path, data_root: Path, name: str) -> dict:
    spec = PILOTS[name]
    payload = json.loads(statistics.read_text(encoding="utf-8"))
    normalization = payload.get("representation", {}).get("normalization", {})
    if normalization.get("version") != "per_horizon_bounds_q99_v1":
        raise ValueError("v30 requires per-horizon bounds statistics")
    config = deepcopy(base)
    data = config["datasets"]["vla_data"]
    data.update(
        {
            "data_root_dir": str(data_root.resolve()),
            "data_mix": spec["data_mix"],
            "window_size": 5,
            "observation_stride": 3,
            "image_aug": True,
            "action_representation": "anchored_relative_pose23_v1",
            "relative_action_statistics": str(statistics.resolve()),
            "active_motion_threshold_m": spec["active_threshold"],
            "inactive_keep_probability": spec["inactive_keep"],
        }
    )
    model = config["framework"]["action_model"]
    model["use_relative_action"] = True
    model["loss_dimension_weights"] = [
        1.0 if 9 <= index < 18 else 0.0 for index in range(23)
    ]
    # Receding-horizon control replans frequently, so near-term waypoints
    # receive more weight while the entire 25-step plan remains supervised.
    model["loss_horizon_weights"] = [
        1.5 - index / 24.0 for index in range(25)
    ]
    trainer = config["trainer"]
    trainer["max_train_steps"] = spec["steps"]
    trainer["num_warmup_steps"] = spec["warmup"]
    trainer["save_interval"] = 1000
    trainer["eval_interval"] = 1000
    trainer["is_resume"] = False
    trainer["learning_rate"]["base"] = spec["learning_rate"]
    trainer["learning_rate"]["action_model"] = spec["learning_rate"]
    config["run_id"] = f"v30-{name}"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in PILOTS:
        output = args.output_dir / f"{name}.yaml"
        output.write_text(
            yaml.safe_dump(
                build(base, args.statistics, args.data_root, name),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(f"V30_CONFIG_READY pilot={name} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
