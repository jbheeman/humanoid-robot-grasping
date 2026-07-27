#!/usr/bin/env python3
"""Build matched absolute-action pilots on achieved future-state datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from object_tracking.vla_target_alignment import FUTURE_STATE_TARGET_V1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future1-data-root", type=Path, required=True)
    parser.add_argument("--future1-statistics", type=Path, required=True)
    parser.add_argument("--future3-data-root", type=Path, required=True)
    parser.add_argument("--future3-statistics", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=750)
    args = parser.parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "physical_robot_authorized": False,
        "variants": {},
    }
    for lag, data_root, statistics in (
        (1, args.future1_data_root, args.future1_statistics),
        (3, args.future3_data_root, args.future3_statistics),
    ):
        stats = json.loads(statistics.read_text(encoding="utf-8"))
        target = stats["representation"]["target_alignment"]
        if (
            target["version"] != FUTURE_STATE_TARGET_V1
            or target["lookahead_frames"] != lag
        ):
            raise ValueError(f"{statistics}: mismatched target alignment")
        config = deepcopy(base)
        data = config["datasets"]["vla_data"]
        data["data_root_dir"] = str(data_root.resolve())
        data["action_representation"] = "absolute_pose23"
        data.pop("relative_action_statistics", None)
        data["target_alignment"] = target
        config["framework"]["action_model"]["use_relative_action"] = False
        trainer = config["trainer"]
        trainer["max_train_steps"] = args.max_train_steps
        trainer["num_warmup_steps"] = min(
            100, max(10, args.max_train_steps // 20)
        )
        trainer["save_interval"] = args.max_train_steps
        trainer["eval_interval"] = args.max_train_steps
        trainer["is_resume"] = False
        trainer["resume_epoch"] = None
        trainer["resume_step"] = None
        name = f"future{lag}-absolute-t1-right9"
        run_id = f"pose23-{name}"
        config["run_id"] = run_id
        path = args.output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        manifest["variants"][name] = {
            "config": str(path.resolve()),
            "run_id": run_id,
            "data_root": str(data_root.resolve()),
            "statistics": str(statistics.resolve()),
            "lookahead_frames": lag,
        }
    output = args.output_dir / "PILOT_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"UNIFOLM_ABSOLUTE_FUTURE_PILOTS_READY output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
