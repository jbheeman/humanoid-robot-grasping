#!/usr/bin/env python3
"""Aggregate non-paused Isaac rollout JSONL into VLA promotion evidence.

The rollout producer must advance physics during the measured policy latency
and emit one record per policy/seed/latency profile. This command never imports
ROS or Unitree transport and never authorizes robot execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_tracking.vla_closed_loop_eval import (
    PromotionConfig,
    evaluate_promotion,
    summarize_rollouts,
)


def read_jsonl(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true", help="use 120-rollout pilot gates")
    args = parser.parse_args()

    config = (
        PromotionConfig(minimum_vla_episodes=120, minimum_latency_slice_episodes=20)
        if args.pilot
        else PromotionConfig()
    )
    summary = summarize_rollouts(read_jsonl(args.rollouts))
    report = {
        **summary,
        "promotion": evaluate_promotion(summary, config),
        "inputs": [str(path) for path in args.rollouts],
        "mode": "pilot" if args.pilot else "final",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["promotion"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
