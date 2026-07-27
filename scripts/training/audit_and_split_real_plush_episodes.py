#!/usr/bin/env python3
"""Audit xr_teleoperate episodes and create disk-light training manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from object_tracking.real_episode_qc import (
    audit_episode,
    grouped_cross_validation_folds,
    grouped_split,
    reject_duplicate_contact_frames,
)


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="mark this collection legacy_audit; never assign it to model selection",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        path for path in args.dataset_root.glob("episode_*") if path.is_dir()
    )
    audits = reject_duplicate_contact_frames(audit_episode(path) for path in episode_dirs)
    accepted = [audit for audit in audits if audit.accepted]
    rejected = [audit for audit in audits if not audit.accepted]
    mode = "legacy_audit" if args.legacy else "insufficient"
    assignments: dict[str, object] = {}
    if not args.legacy and len(accepted) >= 30:
        mode = "train_val_test"
        assignments = grouped_split(accepted)
    elif not args.legacy and len(accepted) >= 20:
        mode = "grouped_cross_validation"
        assignments = grouped_cross_validation_folds(accepted)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_rows = []
    for audit in accepted:
        row = asdict(audit)
        row["evaluation_role"] = mode
        if audit.episode_id in assignments:
            row["split" if mode == "train_val_test" else "fold"] = assignments[
                audit.episode_id
            ]
        accepted_rows.append(row)
    write_jsonl(args.output_dir / "accepted_manifest.jsonl", accepted_rows)
    write_jsonl(
        args.output_dir / "rejected_manifest.jsonl",
        [asdict(audit) for audit in rejected],
    )
    report = {
        "schema_version": 1,
        "dataset_root": str(args.dataset_root),
        "episodes_found": len(audits),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "mode": mode,
        "legacy_data_may_select_models": False,
        "ready_for_final_grouped_split": mode == "train_val_test",
        "real_sampling_weight_sweep": [0.25, 0.50, 0.75],
        "rejection_reasons": dict(
            Counter(reason for audit in rejected for reason in audit.reasons)
        ),
        "condition_counts": dict(
            Counter(audit.condition_id for audit in accepted if audit.condition_id)
        ),
        "session_counts": dict(
            Counter(audit.session_id for audit in accepted if audit.session_id)
        ),
    }
    (args.output_dir / "qc_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
