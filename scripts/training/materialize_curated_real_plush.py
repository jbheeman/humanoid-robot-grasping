#!/usr/bin/env python3
"""Materialize a contact-trimmed real dataset from a QC manifest.

The raw backup is immutable. RGB files are hard-linked when source and output
share a filesystem, so the GB10 keeps both views without duplicating image
storage. Depth and audio are omitted because UniFoLM training does not use them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        records.append(value)
    return records


def materialize_episode(
    record: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    source = Path(record["path"])
    destination = output_root / str(record["episode"])
    frame_range = record.get("training_frame_range")
    if not isinstance(frame_range, dict):
        raise ValueError(f"{source}: missing training frame range")
    start = int(frame_range["start"])
    end = int(frame_range["end_exclusive"])
    payload = json.loads((source / "data.json").read_text(encoding="utf-8"))
    frames = payload["data"][start:end]
    if not frames:
        raise ValueError(f"{source}: empty curated frame range")

    files: list[dict[str, object]] = []
    link_modes: set[str] = set()
    for output_index, frame in enumerate(frames):
        frame["idx"] = output_index
        colors = frame.get("colors", {})
        if "color_0" not in colors:
            raise ValueError(f"{source}: frame {start + output_index} has no RGB")
        relative = Path(str(colors["color_0"]))
        source_image = source / relative
        destination_image = destination / relative
        mode = link_or_copy(source_image, destination_image)
        link_modes.add(mode)
        files.append(
            {
                "path": str(destination_image.relative_to(output_root)),
                "bytes": destination_image.stat().st_size,
                "sha256": sha256(destination_image),
            }
        )
        frame["depths"] = {}
        frame["audios"] = {}

    payload["data"] = frames
    payload["curation"] = {
        "schema_version": 1,
        "source_episode": str(source),
        "source_start_frame": start,
        "source_end_exclusive": end,
        "qc_status": record["status"],
        "qc_reasons": record["reasons"],
        "human_reset_tail_excluded": True,
        "depth_and_audio_omitted": True,
    }
    data_path = destination / "data.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    files.append(
        {
            "path": str(data_path.relative_to(output_root)),
            "bytes": data_path.stat().st_size,
            "sha256": sha256(data_path),
        }
    )
    episode = {
        "episode_id": record["episode_id"],
        "episode": record["episode"],
        "source_path": str(source),
        "curated_path": str(destination),
        "frames": len(frames),
        "link_modes": sorted(link_modes),
        "qc_status": record["status"],
    }
    return episode, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--include-status",
        action="append",
        default=["accepted"],
        help="repeat to include additional statuses; accepted is always included",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")

    allowed = set(args.include_status)
    selected = [
        record
        for record in read_manifest(args.qc_manifest)
        if str(record.get("status")) in allowed
    ]
    if not selected:
        raise SystemExit("QC manifest contains no selected episodes")
    args.output_root.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    files: list[dict[str, object]] = []
    for record in selected:
        episode, episode_files = materialize_episode(record, args.output_root)
        episodes.append(episode)
        files.extend(episode_files)

    manifest = {
        "schema_version": 1,
        "qc_manifest": str(args.qc_manifest),
        "included_statuses": sorted(allowed),
        "episodes": episodes,
        "files": files,
        "raw_backup_modified": False,
    }
    output = args.output_root / "CURATED_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "episodes": len(episodes),
                "files": len(files),
                "bytes": sum(int(item["bytes"]) for item in files),
                "raw_backup_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
