#!/usr/bin/env python3
"""Record immutable inputs for one Isaac interception matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--unitree-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intercept-config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    value = {
        "schema_version": 1,
        "project_commit": git_output(project, "rev-parse", "HEAD"),
        "project_tracked_dirty": bool(
            git_output(project, "status", "--porcelain", "--untracked-files=no")
        ),
        "unitree_sim_commit": git_output(args.unitree_root.resolve(), "rev-parse", "HEAD"),
        "manifest_sha256": sha256_file(args.manifest),
        "intercept_config_sha256": sha256_file(args.intercept_config),
        "runner_sha256": sha256_file(args.runner),
        "planner_sha256": sha256_file(args.planner),
        "planner_urdf_sha256": sha256_file(args.urdf),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
