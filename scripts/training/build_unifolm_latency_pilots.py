#!/usr/bin/env python3
"""Build matched achieved-future-state UniFoLM latency pilot configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from object_tracking.unifolm_relative_actions import RELATIVE_POSE23_V1
from object_tracking.vla_target_alignment import FUTURE_STATE_TARGET_V1


RIGHT_POSE = range(9, 18)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def right_only_loss_weights() -> list[float]:
    return [1.0 if index in RIGHT_POSE else 0.0 for index in range(23)]


def build_config(
    base: dict[str, Any],
    *,
    lookahead: int,
    data_root: Path,
    statistics: Path,
    max_train_steps: int,
) -> dict[str, Any]:
    config = deepcopy(base)
    data = config["datasets"]["vla_data"]
    data["data_root_dir"] = str(data_root.resolve())
    data["window_size"] = 1
    data["observation_stride"] = 1
    data["action_representation"] = RELATIVE_POSE23_V1
    data["relative_action_statistics"] = str(statistics.resolve())
    data["target_alignment"] = {
        "version": FUTURE_STATE_TARGET_V1,
        "lookahead_frames": lookahead,
        "terminal_policy": "drop_unobservable_targets",
    }
    action_model = config["framework"]["action_model"]
    action_model["use_relative_action"] = True
    action_model["loss_dimension_weights"] = right_only_loss_weights()
    trainer = config["trainer"]
    trainer["max_train_steps"] = max_train_steps
    trainer["num_warmup_steps"] = min(
        100, max(10, max_train_steps // 20)
    )
    trainer["save_interval"] = max_train_steps
    trainer["eval_interval"] = max_train_steps
    trainer["learning_rate"]["base"] = 3e-5
    trainer["learning_rate"]["action_model"] = 3e-5
    trainer["is_resume"] = False
    trainer["resume_epoch"] = None
    trainer["resume_step"] = None
    config["run_id"] = f"pose23-latency-future{lookahead}-t1-right9"
    return config


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
    if args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive")

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    candidates = {
        1: (args.future1_data_root, args.future1_statistics),
        3: (args.future3_data_root, args.future3_statistics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "target_version": FUTURE_STATE_TARGET_V1,
        "selection_split": "validation",
        "test_split_sealed": True,
        "physical_robot_authorized": False,
        "variants": {},
    }
    for lookahead, (data_root, statistics) in candidates.items():
        stats_payload = json.loads(statistics.read_text(encoding="utf-8"))
        alignment = stats_payload["representation"]["target_alignment"]
        if (
            alignment.get("version") != FUTURE_STATE_TARGET_V1
            or alignment.get("lookahead_frames") != lookahead
        ):
            raise ValueError(
                f"{statistics}: statistics do not match future{lookahead}"
            )
        config = build_config(
            base,
            lookahead=lookahead,
            data_root=data_root,
            statistics=statistics,
            max_train_steps=args.max_train_steps,
        )
        name = f"future{lookahead}-t1-right9"
        path = args.output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        manifest["variants"][name] = {
            "config": str(path.resolve()),
            "data_root": str(data_root.resolve()),
            "statistics": str(statistics.resolve()),
            "statistics_sha256": sha256(statistics),
            "lookahead_frames": lookahead,
            "max_train_steps": args.max_train_steps,
        }
    output = args.output_dir / "PILOT_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"UNIFOLM_LATENCY_PILOTS_READY output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
