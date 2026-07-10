from __future__ import annotations

import argparse
import json
import os
import signal
import time
from typing import Any
from urllib import error, request

from object_tracking.arm_tracking import AimConfig, aim_from_track, select_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a tracked plushie's image position and aim a G1 arm.")
    parser.add_argument("--vision-url", default="http://127.0.0.1:8000")
    parser.add_argument("--arm-bridge-url", default="http://192.168.0.213:8766")
    parser.add_argument("--token", default=os.environ.get("ARM_TRACKING_TOKEN", ""))
    parser.add_argument("--execute", action="store_true", help="Send commands to the robot arm bridge.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-hz", type=float, default=10.0)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--prediction-s", type=float, default=0.15)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--min-age-frames", type=int, default=3)
    parser.add_argument("--lost-timeout-s", type=float, default=0.5)
    return parser


def read_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], token: str, timeout: float = 2.0) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = build_parser().parse_args()
    if args.execute and not args.token:
        raise SystemExit("--execute requires --token or ARM_TRACKING_TOKEN.")
    if args.poll_hz <= 0 or args.frame_width <= 0 or args.frame_height <= 0:
        raise SystemExit("poll rate and frame dimensions must be positive")

    config = AimConfig(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        prediction_s=args.prediction_s,
        deadzone=args.deadzone,
    )
    stop_requested = False
    preferred_track_id: int | None = None
    last_target_time = 0.0
    stop_sent = False

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_requested
        del signum, frame
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stop_requested:
            loop_started = time.monotonic()
            try:
                payload = read_json(f"{args.vision_url.rstrip('/')}/tracks")
                track = select_target(
                    payload.get("tracks", []),
                    min_confidence=args.min_confidence,
                    min_age_frames=args.min_age_frames,
                    preferred_track_id=preferred_track_id,
                )
                if track is not None:
                    command = aim_from_track(track, config)
                    preferred_track_id = command.track_id
                    last_target_time = time.monotonic()
                    stop_sent = False
                    command_payload = command.as_dict()
                    command_payload["frame_count"] = int(payload.get("frame_count", 0))
                    if args.execute:
                        command_payload["bridge_response"] = post_json(
                            f"{args.arm_bridge_url.rstrip('/')}/target",
                            command_payload,
                            args.token,
                        )
                    else:
                        command_payload["dry_run"] = True
                    print(json.dumps(command_payload, sort_keys=True), flush=True)
                elif time.monotonic() - last_target_time >= args.lost_timeout_s:
                    preferred_track_id = None
                    if args.execute and not stop_sent:
                        response = post_json(
                            f"{args.arm_bridge_url.rstrip('/')}/stop",
                            {"reason": "target_lost"},
                            args.token,
                        )
                        print(json.dumps({"event": "target_lost", "bridge_response": response}), flush=True)
                    stop_sent = True
            except (OSError, ValueError, error.URLError, json.JSONDecodeError) as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), flush=True)

            if args.once:
                break
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, 1.0 / args.poll_hz - elapsed))
    finally:
        if args.execute:
            try:
                post_json(
                    f"{args.arm_bridge_url.rstrip('/')}/stop",
                    {"reason": "client_exit"},
                    args.token,
                )
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
