#!/usr/bin/env python3
"""Build matched translation-focused pilots from the future-1 winner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml


def weights(rotation_weight: float) -> list[float]:
    result = [0.0] * 23
    result[9:12] = [1.0] * 3
    result[12:18] = [rotation_weight] * 6
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=750)
    args = parser.parse_args()
    if args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive")
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    variants = {"xyz3": 0.0, "xyz3-rot01": 0.1}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "physical_robot_authorized": False,
        "variants": {},
    }
    for name, rotation_weight in variants.items():
        config = deepcopy(base)
        config["framework"]["action_model"]["loss_dimension_weights"] = weights(
            rotation_weight
        )
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
        run_id = f"pose23-translation-future1-{name}"
        config["run_id"] = run_id
        path = args.output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        manifest["variants"][name] = {
            "config": str(path.resolve()),
            "run_id": run_id,
            "translation_weight": 1.0,
            "rotation_weight": rotation_weight,
            "max_train_steps": args.max_train_steps,
        }
    output = args.output_dir / "PILOT_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"UNIFOLM_TRANSLATION_PILOTS_READY output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
