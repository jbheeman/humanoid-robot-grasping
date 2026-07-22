#!/usr/bin/env python3
"""Inventory and explicitly prune superseded VLA artifacts.

Dry-run is the default.  Applying a cleanup requires the exact JSON manifest
written by a previous dry-run so no unresolved glob or environment variable is
ever used as a destructive target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


KEEP_CHECKPOINT = "steps_8500_action_model.pt"
EXACT_DELETE_DIRS = (
    "datasets/plush_touch_v5",
    "datasets/plush_touch_rlds_v1/g1_plush_touch_sim",
    "runs/unifolm_plush_touch/plush-touch-tiny",
    "runs/unifolm_plush_touch/plush-touch-smoke",
    "datasets/moving_block_review_v8_rightonly_repair",
)
OBSOLETE_DATASET_PREFIXES = (
    "brainco_contact_diag_v6",
    "brainco_contact_smoke",
    "moving_block_left_hip",
    "moving_block_official_left_rest",
    "moving_block_review_v7",
    "moving_block_rightonly_hidden_smoke",
    "moving_block_rightonly_smoke",
    "moving_block_smoke_v7",
)
ARCHIVE_FILES = (
    "datasets/plush_touch_v5/FINAL_AUDIT.txt",
    "datasets/plush_touch_v5/FINALIZER_STATUS.log",
    "datasets/plush_touch_v5/FINAL_UNIFOLM_SMOKE.txt",
    "datasets/plush_touch_v5/FINAL_MANIFEST.json",
    "runs/unifolm_plush_touch/AUTO_TRAINING_SUMMARY.json",
    "runs/unifolm_plush_touch/AUTOPILOT_STATUS.json",
    "runs/unifolm_plush_touch/heldout_10k.json",
    "runs/unifolm_plush_touch/heldout_selected_test.json",
    "runs/unifolm_plush_touch/selected_merged_vla/MERGE_MANIFEST.json",
)


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_root(root: Path) -> Path:
    root = root.resolve()
    required = (root / "pyproject.toml", root / "scripts", root / "datasets")
    if not all(item.exists() for item in required):
        raise SystemExit(f"refusing unknown project root: {root}")
    if root == Path("/") or len(root.parts) < 4:
        raise SystemExit(f"refusing broad project root: {root}")
    return root


def resolved_candidates(root: Path) -> list[Path]:
    candidates = [root / relative for relative in EXACT_DELETE_DIRS]
    dataset_root = root / "datasets"
    for child in dataset_root.iterdir():
        if any(child.name.startswith(prefix) for prefix in OBSOLETE_DATASET_PREFIXES):
            candidates.append(child)
    checkpoint_root = root / "runs/unifolm_plush_touch/plush-touch-full/checkpoints"
    if checkpoint_root.is_dir():
        candidates.extend(
            child for child in checkpoint_root.glob("steps_*_action_model.pt")
            if child.name != KEEP_CHECKPOINT
        )
    unique = sorted({path.resolve() for path in candidates if path.exists()})
    allowed_roots = ((root / "datasets").resolve(), (root / "runs").resolve())
    for path in unique:
        if not any(path.is_relative_to(parent) for parent in allowed_roots):
            raise SystemExit(f"candidate escaped allowed roots: {path}")
    return unique


def archive_metadata(root: Path, archive: Path) -> list[dict[str, object]]:
    copied = []
    for relative in ARCHIVE_FILES:
        source = root / relative
        if not source.is_file():
            continue
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {"source": relative, "archive": str(destination.relative_to(root)),
             "bytes": source.stat().st_size, "sha256": sha256(source)}
        )
    return copied


def build_manifest(root: Path) -> dict[str, object]:
    candidates = resolved_candidates(root)
    selected = root / "runs/unifolm_plush_touch/selected_merged_vla/pytorch_model.pt"
    kept_head = root / "runs/unifolm_plush_touch/plush-touch-full/checkpoints" / KEEP_CHECKPOINT
    return {
        "schema": 1,
        "created_unix": time.time(),
        "project_root": str(root),
        "free_bytes_before": shutil.disk_usage(root).free,
        "protected": [
            str(selected.relative_to(root)),
            str(kept_head.relative_to(root)),
            "models/pretrained",
            "datasets/plush_touch_real_raw",
            "datasets/plush_touch_canonical_v1",
            "datasets/plush_touch_rlds_v1/g1_plush_touch_real",
            "datasets/moving_block_review_v8_rightonly",
            "artifacts/vla_dataset_review",
        ],
        "selected_model": {
            "path": str(selected.relative_to(root)),
            "bytes": selected.stat().st_size if selected.is_file() else None,
        },
        "delete": [
            {"path": str(path.relative_to(root)), "bytes": tree_size(path)}
            for path in candidates
        ],
        "recoverable_bytes": sum(tree_size(path) for path in candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = validate_root(args.root)
    archive_root = root / "artifacts/archive/storage"
    archive_root.mkdir(parents=True, exist_ok=True)

    if not args.apply:
        manifest = build_manifest(root)
        output = args.manifest or archive_root / "retention_manifest.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "mode": "dry-run", "manifest": str(output),
            "targets": len(manifest["delete"]),
            "recoverable_gib": round(manifest["recoverable_bytes"] / 1024**3, 2),
            "free_gib": round(manifest["free_bytes_before"] / 1024**3, 2),
        }, sort_keys=True))
        for item in manifest["delete"]:
            print(f"DELETE {item['bytes'] / 1024**3:8.2f} GiB  {item['path']}")
        return 0

    if args.manifest is None or not args.manifest.is_file():
        raise SystemExit("--apply requires an existing --manifest from dry-run")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    current = build_manifest(root)
    planned = [(item["path"], item["bytes"]) for item in manifest.get("delete", [])]
    observed = [(item["path"], item["bytes"]) for item in current.get("delete", [])]
    if manifest.get("project_root") != str(root) or planned != observed:
        raise SystemExit("cleanup targets changed since dry-run; generate and review a new manifest")

    archived = archive_metadata(root, archive_root / "retained_metadata")
    for item in manifest["delete"]:
        target = (root / item["path"]).resolve()
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        print(f"REMOVED {item['path']}")
    result = {
        "applied_unix": time.time(), "manifest": str(args.manifest),
        "archived": archived, "free_bytes_after": shutil.disk_usage(root).free,
    }
    result_path = archive_root / "cleanup_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": "applied", "removed": len(manifest["delete"]),
        "free_gib": round(result["free_bytes_after"] / 1024**3, 2),
        "result": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
