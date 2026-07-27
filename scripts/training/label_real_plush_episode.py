#!/usr/bin/env python3
"""Write the companion metadata required by real bunny episode QC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--block", required=True, help="repeat group; kept in one split")
    parser.add_argument("--path-angle-deg", required=True, type=float)
    parser.add_argument("--speed-m-s", required=True, type=float)
    parser.add_argument("--arm-start", required=True)
    parser.add_argument("--camera-view", default="head")
    parser.add_argument("--visual-contact-frame", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_path = args.episode / "data.json"
    if not data_path.is_file():
        raise SystemExit(f"missing episode data: {data_path}")
    output = args.episode / "episode_metadata.json"
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force after review")
    if args.speed_m_s <= 0.0:
        raise SystemExit("--speed-m-s must be positive")
    if args.visual_contact_frame < 0:
        raise SystemExit("--visual-contact-frame cannot be negative")
    payload = {
        "schema_version": 1,
        "collection_session_id": args.session,
        "collection_block_id": args.block,
        "rabbit_path_angle_deg": args.path_angle_deg,
        "rabbit_speed_m_s": args.speed_m_s,
        "right_arm_start": args.arm_start,
        "camera_view": args.camera_view,
        "moving_object": True,
        "contact_reviewed": True,
        "visual_contact_frame": args.visual_contact_frame,
    }
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
