"""Evaluate bunny future-position predictions from offline JSONL tracks."""
from __future__ import annotations

import argparse
from pathlib import Path

from .future_prediction import evaluate_future_positions, load_observations, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path, help="JSONL observations from video/depth replay")
    parser.add_argument("--output", type=Path, default=Path("runs/offline/future_prediction"))
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args(argv)
    report = evaluate_future_positions(load_observations(args.tracks), min_confidence=args.min_confidence)
    json_path, markdown_path = write_report(report, args.output)
    print(f"report: {json_path}\nsummary: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
