#!/usr/bin/env python3
"""Aggregate physics-advanced Isaac interception rollouts.

The producer must run Isaac actuator dynamics during perception and command
latency.  This command is offline-only: it reads JSONL, emits a report, and
never imports ROS, Unitree transport, or the Isaac runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_tracking.intercept_closed_loop_eval import (
    InterceptPromotionConfig,
    evaluate_intercept_promotion,
    summarize_intercept_rollouts,
)


def read_jsonl(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--candidate", choices=("cv_ik", "alpha_beta_ik", "gru_ik"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-scenarios-per-slice",
        type=int,
        default=30,
        help="use at least 50 per slice for a final rather than pilot decision",
    )
    args = parser.parse_args()
    if args.minimum_scenarios_per_slice <= 0:
        raise SystemExit("--minimum-scenarios-per-slice must be positive")

    summary = summarize_intercept_rollouts(read_jsonl(args.rollouts))
    promotion = evaluate_intercept_promotion(
        summary,
        candidate_policy=args.candidate,
        config=InterceptPromotionConfig(
            minimum_scenarios_per_slice=args.minimum_scenarios_per_slice
        ),
    )
    report = {
        **summary,
        "promotion": promotion,
        "inputs": [str(path) for path in args.rollouts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(promotion, sort_keys=True))
    return 0 if promotion["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
