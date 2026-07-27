#!/usr/bin/env python3
"""Export a GB10 research session into the portable Isaac replay contract."""

from __future__ import annotations

import argparse
import json

from object_tracking.arm_tracking.sim_validation import (
    build_replay_episode,
    write_replay_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", help="GB10 telemetry.jsonl")
    parser.add_argument("output", help="destination replay JSON")
    parser.add_argument(
        "--episode",
        type=int,
        default=-1,
        help="detected target episode index; negative indices count from the end",
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        help="explicit session_elapsed_s window instead of automatic episode detection",
    )
    args = parser.parse_args()
    episode = build_replay_episode(
        args.telemetry,
        episode_index=args.episode,
        window=None if args.window is None else tuple(args.window),
    )
    write_replay_episode(episode, args.output)
    print(json.dumps(episode.to_dict()["diagnostics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

