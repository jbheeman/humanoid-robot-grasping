#!/usr/bin/env python3
"""Apply explicit human decisions to needs-review episode QC records."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def apply_decisions(
    records: list[dict[str, Any]],
    decisions: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    unresolved: list[int] = []
    seen: set[int] = set()
    for original in records:
        record = dict(original)
        episode_id = int(record["episode_id"])
        if record["status"] == "needs_review":
            decision = decisions.get(episode_id)
            if decision is None:
                unresolved.append(episode_id)
            else:
                status = decision["status"]
                if status not in ("accepted", "automatic_reject"):
                    raise ValueError(
                        f"episode {episode_id}: invalid review status {status}"
                    )
                record["automatic_status"] = "needs_review"
                record["automatic_reasons"] = list(record["reasons"])
                record["status"] = status
                record["reasons"] = (
                    [] if status == "accepted" else [decision["reason"]]
                )
                record["human_review"] = {
                    "decision": status,
                    "reason": decision["reason"],
                }
                seen.add(episode_id)
        output.append(record)
    unknown = sorted(set(decisions) - seen)
    if unknown:
        raise ValueError(f"decisions do not match needs-review episodes: {unknown}")
    if unresolved:
        raise ValueError(f"needs-review episodes remain unresolved: {unresolved}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise SystemExit(f"refusing to overwrite {args.output_manifest}")
    payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = {
        int(episode_id): value
        for episode_id, value in payload["episodes"].items()
    }
    records = apply_decisions(read_jsonl(args.input_manifest), decisions)
    args.output_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_manifest),
                "episodes": len(records),
                "status_counts": dict(Counter(r["status"] for r in records)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
