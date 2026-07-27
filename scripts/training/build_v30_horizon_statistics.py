#!/usr/bin/env python3
"""Materialize 25-waypoint normalization arrays from audited v29 statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


FIELDS = ("min", "max", "mean", "std", "q01", "q99")


def build(source: dict[str, Any], horizon: int) -> dict[str, Any]:
    per_horizon = source["provenance"]["per_horizon"]
    missing = [str(index) for index in range(horizon) if str(index) not in per_horizon]
    if missing:
        raise ValueError(f"missing per-horizon statistics: {missing}")
    result = json.loads(json.dumps(source))
    result["horizon_action"] = {
        field: [
            per_horizon[str(index)]["action"][field] for index in range(horizon)
        ]
        for field in FIELDS
    }
    representation = result["representation"]
    if int(representation["horizon"]) != horizon:
        raise ValueError("representation horizon mismatch")
    representation["normalization"] = {
        "version": "per_horizon_bounds_q99_v1",
        "horizon": horizon,
        "source": "provenance.per_horizon",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=25)
    args = parser.parse_args()
    payload = build(json.loads(args.input.read_text(encoding="utf-8")), args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(f"V30_HORIZON_STATS_READY output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
