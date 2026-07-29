#!/usr/bin/env python3
"""Fail-fast audit for the first episode of an Isaac interception matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()

    shared = runpy.run_path(
        str(Path(__file__).with_name("summarize-intercept-matrix.py"))
    )
    value = json.loads(args.result.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    failures = shared["audit_result"](
        value,
        provenance,
        expected_replay_sha256=shared["sha256_file"](args.scenario),
    )
    report = {
        "schema_version": 1,
        "passed": not failures,
        "result": str(args.result),
        "scenario": str(args.scenario),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
