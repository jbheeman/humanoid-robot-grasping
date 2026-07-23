#!/usr/bin/env python3
"""Replay timestamped VLA/tracker records through the production safety gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_tracking.vla_chunk_scheduler import ScheduleDecision, VLAChunkScheduler
from object_tracking.vla_visual_safety import VisualSafetyGovernor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="chronological JSONL replay")
    parser.add_argument("--action-dimension", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-unsafe-allows", type=int, default=0)
    args = parser.parse_args()

    scheduler = VLAChunkScheduler(args.action_dimension)
    governor = VisualSafetyGovernor()
    counts = {decision.value: 0 for decision in ScheduleDecision}
    reasons: dict[str, int] = {}
    unsafe_allows = 0
    expired_waypoints = 0
    executed_waypoint_indices: list[int] = []
    previous_time = float("-inf")
    records = 0
    for line in args.replay.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        now_s = float(record["timestamp_s"])
        if now_s <= previous_time:
            raise ValueError("replay timestamps must be strictly increasing")
        previous_time = now_s
        if "actions" in record:
            result = scheduler.submit(
                record["actions"],
                observation_time_s=float(record["observation_time_s"]),
                inference_completed_time_s=float(record["inference_completed_time_s"]),
                now_s=now_s,
                state_anchor_time_s=(
                    float(record["state_anchor_time_s"])
                    if "state_anchor_time_s" in record
                    else None
                ),
                inference_started_time_s=(
                    float(record["inference_started_time_s"])
                    if "inference_started_time_s" in record
                    else None
                ),
            )
            counts[result.decision.value] += 1
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            expired_waypoints += result.expired_waypoints
        signal = governor.signal(
            record.get("tracks", []),
            now_s=now_s,
            table_clearance_m=float(record.get("table_clearance_m", float("inf"))),
            contact=bool(record.get("contact", False)),
        )
        result = scheduler.tick(now_s=now_s, safety=signal)
        counts[result.decision.value] += 1
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        expired_waypoints += result.expired_waypoints
        if result.waypoint_index is not None:
            executed_waypoint_indices.append(result.waypoint_index)
        if result.decision is ScheduleDecision.EXECUTE and bool(record.get("unsafe", False)):
            unsafe_allows += 1
        records += 1

    report = {
        "records": records,
        "decisions": counts,
        "reasons": reasons,
        "unsafe_allows": unsafe_allows,
        "expired_waypoints": expired_waypoints,
        "executed_waypoint_indices": executed_waypoint_indices,
        "passed": unsafe_allows <= args.maximum_unsafe_allows,
        "causal": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
