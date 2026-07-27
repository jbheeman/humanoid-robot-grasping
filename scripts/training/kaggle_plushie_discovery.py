#!/usr/bin/env python3
"""Create an auditable Kaggle candidate list without silently contaminating labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


QUERIES = ("plush toy yolo", "teddy bear object detection", "stuffed animal detection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/training_logs/kaggle_candidates.json"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for query in QUERIES:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--search", query, "--csv"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(result.stderr.strip() or "Kaggle search failed")
        for row in csv.DictReader(result.stdout.splitlines()):
            rows.append({"query": query, **row})
    unique = {row.get("ref", ""): row for row in rows if row.get("ref")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sorted(unique.values(), key=lambda row: row["ref"]), indent=2) + "\n")
    print(json.dumps({"candidates": len(unique), "output": str(args.output)}))


if __name__ == "__main__":
    main()
