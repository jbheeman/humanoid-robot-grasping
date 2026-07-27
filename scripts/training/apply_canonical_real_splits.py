#!/usr/bin/env python3
"""Apply an explicit, leakage-safe split manifest to canonical real episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


VALID_SPLITS = frozenset({"train", "validation", "test"})


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_splits(canonical_root: Path, split_manifest: Path) -> dict[str, int]:
    requested = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignments = requested["episodes"]
    invalid = sorted(set(assignments.values()) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"invalid splits: {invalid}")

    manifest_path = canonical_root / "CANONICAL_MANIFEST.json"
    canonical = json.loads(manifest_path.read_text(encoding="utf-8"))
    real_records = [
        record for record in canonical["episodes"] if record["source"] == "xr_teleoperate"
    ]
    episode_records = {Path(record["path"]).stem: record for record in real_records}
    missing = sorted(set(episode_records) - set(assignments))
    extra = sorted(set(assignments) - set(episode_records))
    if missing or extra:
        raise ValueError(f"split coverage mismatch: missing={missing}, extra={extra}")

    counts = {split: 0 for split in sorted(VALID_SPLITS)}
    for episode, record in sorted(episode_records.items()):
        split = assignments[episode]
        source = Path(record["path"])
        destination = canonical_root / "hdf5" / "real" / split / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            if destination.exists():
                raise FileExistsError(destination)
            source.replace(destination)
        with h5py.File(destination, "r+") as output:
            output.attrs["split"] = split
        record["path"] = str(destination)
        record["split"] = split
        counts[split] += 1

    canonical["real_split_manifest"] = str(split_manifest.resolve())
    canonical["real_split_strategy"] = requested.get("strategy", "explicit")
    _write_json_atomic(manifest_path, canonical)
    _write_json_atomic(canonical_root / "REAL_SPLITS.json", requested)

    for directory in (canonical_root / "hdf5" / "real").iterdir():
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    counts = apply_splits(args.canonical_root, args.split_manifest)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
