#!/usr/bin/env python3
"""Wait for a remote job process to exit without polling or sleeping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select


def tail(path: Path | None, lines: int = 30) -> list[str]:
    if path is None or not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]


def marker_is_complete(marker_text: str) -> bool:
    """Accept either an atomic JSON status file or a simple shell marker."""
    if not marker_text:
        return False
    try:
        value = json.loads(marker_text)
    except json.JSONDecodeError:
        return "complete=1" in marker_text
    return isinstance(value, dict) and value.get("complete") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, action="append", required=True)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if any(pid <= 1 for pid in args.pid):
        raise SystemExit("refusing to monitor an unsafe PID")

    # pidfd is a kernel event source: poll blocks until the exact process exits,
    # avoiding PID-name races and repeated `sleep`/`ps` checks.
    events = select.poll()
    pidfds: dict[int, int] = {}
    for pid in args.pid:
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        pidfds[pidfd] = pid
        events.register(pidfd, select.POLLIN)
    try:
        while pidfds:
            for pidfd, _event in events.poll():
                events.unregister(pidfd)
                os.close(pidfd)
                del pidfds[pidfd]
    finally:
        for pidfd in pidfds:
            os.close(pidfd)

    marker_text = (
        args.marker.read_text(errors="replace")
        if args.marker is not None and args.marker.is_file()
        else ""
    )
    complete = args.marker is None or marker_is_complete(marker_text)
    payload = {
        "complete": complete,
        "marker": None if args.marker is None else str(args.marker),
        "marker_text": marker_text,
        "pids": args.pid,
        "log_tail": tail(args.log),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
