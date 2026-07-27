#!/usr/bin/env python3
"""Extract train-only real bunny frames for sim/real visual calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def contact_indices(payload: dict) -> list[int]:
    result = []
    for index, frame in enumerate(payload.get("data", [])):
        annotation = frame.get("annotations", {}).get("right_palm_contact", {})
        if bool(annotation.get("value", False)):
            result.append(index)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    split_payload = json.loads(args.splits.read_text())
    train_names = sorted(
        name for name, split in split_payload["episodes"].items() if split == "train"
    )
    candidates: list[tuple[str, int, Path]] = []
    for episode_name in train_names:
        episode = args.raw_root / episode_name
        payload = json.loads((episode / "data.json").read_text())
        indices = contact_indices(payload)
        if not indices:
            continue
        selected = [indices[0]]
        # Add a late-approach frame as a second view when 10 train episodes
        # need to supply a 12-pair audit. It remains strictly train-only.
        selected.append(max(0, indices[0] - 12))
        for index in selected:
            frame = payload["data"][index]
            relative = frame["colors"]["color_0"]
            candidates.append((episode_name, index, episode / relative))
    if len(candidates) < args.count:
        raise SystemExit(f"only {len(candidates)} train-only calibration frames are available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    records = []
    for episode_name, index, source in candidates[: args.count]:
        output = args.output_dir / f"{episode_name}_frame_{index:06d}.jpg"
        shutil.copy2(source, output)
        manifest[output.name] = "train"
        records.append({"episode": episode_name, "frame": index, "source": str(source)})
    (args.output_dir / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"frames": len(records), "split": "train"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
