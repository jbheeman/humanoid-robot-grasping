#!/usr/bin/env python3
"""Direct Unitree SDK2 plushie tracking test (no ROS).

This script subscribes to the GB10 `/tracks` endpoint and applies bounded
shoulder-joint corrections in the native SDK2 arm command path.  The head
camera is fixed to the body, so 2-D image coordinates are *not* closed-loop
arm feedback: moving the arm cannot center a detection in that camera.  This
therefore uses a guarded visual-lock pose around the measured start pose rather
than an unbounded image-error integrator.  It is intentionally separate from
the ROS2 commissioning stack and should be run only when no other process is
publishing to `rt/arm_sdk`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
for candidate in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from object_tracking.arm_tracking.arm_bridge import ArmCommand, RobotState
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware
from object_tracking.arm_tracking.joints import DEFAULT_RIGHT_JOINT_LIMITS


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _now() -> float:
    return time.time()


def _parse_joint_names(raw: str) -> list[str]:
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        return []
    return names


def _fetch_tracks(url: str, timeout_s: float = 0.75) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    tracks = payload.get("tracks") if isinstance(payload, dict) else None
    return tracks if isinstance(tracks, list) else []


@dataclass
class TrackerConfig:
    tracks_url: str
    frame_width: float
    frame_height: float
    refresh_hz: float = 40.0
    control_hz: float = 50.0
    min_confidence: float = 0.55
    gain_x: float = 0.10
    gain_y: float = 0.10
    max_step_rad: float = 0.015
    filter_alpha: float = 0.55
    wanted_classes: list[str] | None = None
    hold_seconds: float = 0.6
    interface: str = "eth0"
    domain_id: int = 0
    startup_raise_rad: float = 0.10
    startup_raise_seconds: float = 1.0
    reverse_x: bool = False
    reverse_y: bool = False
    slew_step_rad: float = 0.003
    aim_x_px: float = 480.0
    aim_y_px: float = 335.0
    deadband_px: float = 35.0
    pitch_envelope_rad: float = 0.12
    yaw_envelope_rad: float = 0.16
    lock_frames: int = 6
    # Match Unitree's official G1 arm-SDK DDS example.  These are not
    # simulation-derived gains; motion is bounded by the visual-lock envelope
    # and slew limiter below.
    kp: float = 60.0
    kd: float = 1.5

    def __post_init__(self) -> None:
        if self.refresh_hz <= 0:
            raise ValueError("refresh_hz must be > 0")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be > 0")
        if not (0 < self.min_confidence <= 1):
            raise ValueError("min_confidence must be in (0, 1]")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame width/height must be positive")


def _select_track(
    tracks: list[dict[str, Any]],
    min_confidence: float,
    wanted: list[str] | None,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        confidence = float(track.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        class_name = str(track.get("class_name", "")).strip().lower()
        if wanted and not any(token in class_name for token in wanted):
            continue
        bbox = track.get("bbox_xyxy")
        center = track.get("center_xy")
        if not isinstance(bbox, (list, tuple)) or not isinstance(center, (list, tuple)):
            continue
        if len(center) != 2 or len(bbox) != 4:
            continue
        candidates.append((confidence, track))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


@dataclass(frozen=True)
class TrackSnapshot:
    sequence: int
    received_at: float
    tracks: list[dict[str, Any]]


class TrackPoller:
    """Keep HTTP latency out of the 50 Hz native arm-SDK command loop."""

    def __init__(self, url: str, rate_hz: float) -> None:
        self._url = url
        self._period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot: TrackSnapshot | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="plushie-track-poller")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def snapshot(self) -> TrackSnapshot | None:
        with self._lock:
            return self._snapshot

    def _run(self) -> None:
        sequence = 0
        while not self._stop.is_set():
            started = _now()
            try:
                tracks = _fetch_tracks(self._url, timeout_s=min(0.5, self._period * 2.0))
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                tracks = []
            sequence += 1
            with self._lock:
                self._snapshot = TrackSnapshot(sequence, _now(), tracks)
            self._stop.wait(max(0.0, self._period - (_now() - started)))


def _extract_center(track: dict[str, Any]) -> tuple[float, float]:
    center = track.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return float(center[0]), float(center[1])
    bbox = track.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("track has no center or bbox")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _make_command(
    state: RobotState,
    shoulder_pitch: float,
    shoulder_roll: float,
    shoulder_yaw: float,
    *,
    kp_value: float,
    kd_value: float,
) -> ArmCommand:
    right_limits = DEFAULT_RIGHT_JOINT_LIMITS
    q = list(state.arm_q)
    # Unitree arm order in this bridge is [left(7), right(7)] in arm_q.
    q[7] = _clamp(
        shoulder_pitch,
        right_limits[0][0],
        right_limits[0][1],
    )
    q[8] = _clamp(
        shoulder_roll,
        right_limits[1][0],
        right_limits[1][1],
    )
    q[9] = _clamp(
        shoulder_yaw,
        right_limits[2][0],
        right_limits[2][1],
    )
    zero = tuple(0.0 for _ in q)
    kp = tuple(kp_value for _ in q)
    kd = tuple(kd_value for _ in q)
    return ArmCommand(
        q=tuple(q),
        dq=zero,
        kp=kp,
        kd=kd,
        weight=1.0,
        mode_machine=int(state.mode_machine),
        published_at=_now(),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Track plushie by tracks endpoint and drive shoulder joints")
    parser.add_argument("--tracks-url", default="http://192.168.0.66:8000/tracks")
    parser.add_argument("--width", type=float, default=960)
    parser.add_argument("--height", type=float, default=540)
    parser.add_argument(
        "--rate",
        type=float,
        default=float(os.environ.get("TRACK_RATE", "40")),
        help="GB10 track-poll rate; keep at or below measured YOLO FPS",
    )
    parser.add_argument(
        "--control-rate",
        type=float,
        default=float(os.environ.get("TRACK_CONTROL_RATE", "50")),
        help="native arm-SDK publish rate; Unitree's reference example uses 50 Hz",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.environ.get("TRACK_MIN_CONFIDENCE", "0.55")),
    )
    parser.add_argument("--gain-x", type=float, default=float(os.environ.get("TRACK_GAIN_X", "0.10")))
    parser.add_argument("--gain-y", type=float, default=float(os.environ.get("TRACK_GAIN_Y", "0.10")))
    parser.add_argument("--max-step-rad", type=float, default=0.015)
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=float(os.environ.get("TRACK_FILTER_ALPHA", "0.18")),
    )
    parser.add_argument("--hold-seconds", type=float, default=0.6)
    parser.add_argument(
        "--class",
        dest="classes",
        default="plush,bunny",
        help="comma-separated allowed class_name tokens",
    )
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--reverse-x",
        action="store_true",
        default=os.environ.get("TRACK_REVERSE_X", "0") == "1",
    )
    parser.add_argument(
        "--reverse-y",
        action="store_true",
        default=os.environ.get("TRACK_REVERSE_Y", "0") == "1",
    )
    parser.add_argument(
        "--slew-step-rad",
        type=float,
        default=float(os.environ.get("TRACK_SLEW_STEP_RAD", "0.003")),
        help="hard per-update joint limit used to reduce jerk",
    )
    parser.add_argument("--aim-x-px", type=float, default=float(os.environ.get("TRACK_AIM_X_PX", "480")))
    parser.add_argument("--aim-y-px", type=float, default=float(os.environ.get("TRACK_AIM_Y_PX", "335")))
    parser.add_argument("--deadband-px", type=float, default=float(os.environ.get("TRACK_DEADBAND_PX", "35")))
    parser.add_argument(
        "--pitch-envelope-rad",
        type=float,
        default=float(os.environ.get("TRACK_PITCH_ENVELOPE_RAD", "0.12")),
    )
    parser.add_argument(
        "--yaw-envelope-rad",
        type=float,
        default=float(os.environ.get("TRACK_YAW_ENVELOPE_RAD", "0.16")),
    )
    parser.add_argument("--lock-frames", type=int, default=int(os.environ.get("TRACK_LOCK_FRAMES", "6")))
    parser.add_argument("--kp", type=float, default=float(os.environ.get("TRACK_KP", "60")))
    parser.add_argument("--kd", type=float, default=float(os.environ.get("TRACK_KD", "1.5")))
    parser.add_argument(
        "--startup-raise-rad",
        type=float,
        default=float(os.environ.get("TRACK_STARTUP_RAISE_RAD", "0.10")),
        help="initial right-shoulder-pitch lift used as a tracking-start indicator",
    )
    parser.add_argument(
        "--startup-raise-seconds",
        type=float,
        default=float(os.environ.get("TRACK_STARTUP_RAISE_SECONDS", "1.0")),
    )

    args = parser.parse_args()
    cfg = TrackerConfig(
        tracks_url=args.tracks_url,
        frame_width=args.width,
        frame_height=args.height,
        refresh_hz=args.rate,
        control_hz=args.control_rate,
        min_confidence=args.min_confidence,
        gain_x=args.gain_x,
        gain_y=args.gain_y,
        max_step_rad=args.max_step_rad,
        filter_alpha=args.filter_alpha,
        wanted_classes=_parse_joint_names(args.classes),
        hold_seconds=args.hold_seconds,
        interface=args.interface,
        domain_id=args.domain_id,
        startup_raise_rad=args.startup_raise_rad,
        startup_raise_seconds=max(0.1, args.startup_raise_seconds),
        reverse_x=args.reverse_x,
        reverse_y=args.reverse_y,
        slew_step_rad=max(0.001, min(abs(args.slew_step_rad), abs(args.max_step_rad))),
        aim_x_px=args.aim_x_px,
        aim_y_px=args.aim_y_px,
        deadband_px=max(0.0, args.deadband_px),
        pitch_envelope_rad=max(0.01, abs(args.pitch_envelope_rad)),
        yaw_envelope_rad=max(0.01, abs(args.yaw_envelope_rad)),
        lock_frames=max(1, args.lock_frames),
        kp=max(0.0, args.kp),
        kd=max(0.0, args.kd),
    )

    # Point SDK2 import path at the robot checkout if a standard path exists.
    default_sdk_path = Path.home() / "unitree_sdk2_python"
    if "UNITREE_SDK_PYTHONPATH" not in os.environ and default_sdk_path.exists():
        os.environ["UNITREE_SDK_PYTHONPATH"] = str(default_sdk_path)
    if (p := os.environ.get("UNITREE_SDK_PYTHONPATH", "")).strip():
        existing = os.environ.get("PYTHONPATH", "")
        if p not in existing.split(":"):
            os.environ["PYTHONPATH"] = f"{p}:{existing}" if existing else p

    print("Direct SDK2 tracking will use interface {}, domain {}".format(cfg.interface, cfg.domain_id))
    hardware = UnitreeArmHardware(interface=cfg.interface, domain_id=cfg.domain_id)
    hardware.start()

    # Give a visible, bounded indication that tracking has started.  This is
    # deliberately a small shoulder-only ramp; the remaining arm joints stay
    # at their measured positions.  Set TRACK_STARTUP_RAISE_RAD=0 to disable.
    startup_state = None
    startup_deadline = _now() + 3.0
    while startup_state is None and _now() < startup_deadline:
        startup_state = hardware.latest_state()
        if startup_state is None:
            time.sleep(0.05)
    if startup_state is None:
        raise RuntimeError("No native lowstate received; refusing startup raise")

    target_q = list(startup_state.arm_q[7:])
    if abs(cfg.startup_raise_rad) > 1e-6:
        start_pitch = float(startup_state.arm_q[7])
        end_pitch = _clamp(
            start_pitch + cfg.startup_raise_rad,
            DEFAULT_RIGHT_JOINT_LIMITS[0][0],
            DEFAULT_RIGHT_JOINT_LIMITS[0][1],
        )
        steps = max(1, int(round(cfg.startup_raise_seconds * cfg.refresh_hz)))
        print(
            f"tracking start indicator: shoulder pitch {start_pitch:.3f} -> {end_pitch:.3f} rad",
            flush=True,
        )
        for step in range(1, steps + 1):
            current = hardware.latest_state() or startup_state
            fraction = step / steps
            pitch = start_pitch + (end_pitch - start_pitch) * fraction
            hardware.publish(
                _make_command(
                    current,
                    pitch,
                    current.arm_q[8],
                    current.arm_q[9],
                    kp_value=cfg.kp,
                    kd_value=cfg.kd,
                )
            )
            time.sleep(cfg.startup_raise_seconds / steps)
        target_q[0] = end_pitch

    period = 1.0 / cfg.control_hz
    last_seen = _now()
    smoothed_center: tuple[float, float] | None = None
    visual_lock_q = list(target_q)
    desired_q = list(target_q)
    locked_track_id: object | None = None
    stable_frames = 0
    released = False
    last_snapshot_sequence = -1
    tracking_active = False
    poller = TrackPoller(cfg.tracks_url, cfg.refresh_hz)
    poller.start()

    print("Direct plushie tracker running (ETH0). Press Ctrl-C to stop.")
    print(f"Tracks source: {cfg.tracks_url}; poll={cfg.refresh_hz:.0f} Hz, arm SDK={cfg.control_hz:.0f} Hz")

    try:
        while True:
            loop_start = _now()
            state = hardware.latest_state()
            if state is None:
                time.sleep(0.05)
                continue

            now = _now()
            snapshot = poller.snapshot()
            if snapshot is not None and snapshot.sequence != last_snapshot_sequence:
                last_snapshot_sequence = snapshot.sequence
                track = _select_track(snapshot.tracks, cfg.min_confidence, cfg.wanted_classes)
                if track is None:
                    stable_frames = 0
                    smoothed_center = None
                    locked_track_id = None
                    tracking_active = False
                else:
                    last_seen = snapshot.received_at
                    released = False
                    center_x, center_y = _extract_center(track)
                    if smoothed_center is None:
                        smoothed_center = (center_x, center_y)
                    else:
                        a = min(cfg.filter_alpha, 0.25)
                        smoothed_center = (
                            smoothed_center[0] * (1.0 - a) + center_x * a,
                            smoothed_center[1] * (1.0 - a) + center_y * a,
                        )

                    track_id = track.get("track_id")
                    if track_id != locked_track_id:
                        locked_track_id = track_id
                        stable_frames = 0
                        visual_lock_q = list(state.arm_q[7:])
                    stable_frames += 1
                    if stable_frames < cfg.lock_frames:
                        print(f"locking target {track_id!r}: {stable_frames}/{cfg.lock_frames}", flush=True)
                    else:
                        cx, cy = smoothed_center
                        err_x_px = cx - cfg.aim_x_px
                        err_y_px = cy - cfg.aim_y_px

                        def deadband(error_px: float) -> float:
                            if abs(error_px) <= cfg.deadband_px:
                                return 0.0
                            return error_px - (cfg.deadband_px if error_px > 0 else -cfg.deadband_px)

                        err_x = deadband(err_x_px) / (cfg.frame_width * 0.5)
                        err_y = deadband(err_y_px) / (cfg.frame_height * 0.5)
                        yaw_delta = -err_x * cfg.gain_x
                        pitch_delta = err_y * cfg.gain_y
                        if cfg.reverse_x:
                            yaw_delta = -yaw_delta
                        if cfg.reverse_y:
                            pitch_delta = -pitch_delta

                        desired_q[0] = _clamp(
                            visual_lock_q[0] + _clamp(pitch_delta, -cfg.pitch_envelope_rad, cfg.pitch_envelope_rad),
                            visual_lock_q[0] - cfg.pitch_envelope_rad,
                            visual_lock_q[0] + cfg.pitch_envelope_rad,
                        )
                        desired_q[1] = visual_lock_q[1]
                        desired_q[2] = _clamp(
                            visual_lock_q[2] + _clamp(yaw_delta, -cfg.yaw_envelope_rad, cfg.yaw_envelope_rad),
                            visual_lock_q[2] - cfg.yaw_envelope_rad,
                            visual_lock_q[2] + cfg.yaw_envelope_rad,
                        )
                        tracking_active = True
                        conf = float(track.get("confidence", 0.0) or 0.0)
                        print(
                            f"track={track_id} cls={track.get('class_name')} conf={conf:.2f} "
                            f"cx={cx:.1f} cy={cy:.1f} aim=({cfg.aim_x_px:.0f},{cfg.aim_y_px:.0f}) "
                            f"target_pitch={desired_q[0]:.3f} target_yaw={desired_q[2]:.3f}",
                            flush=True,
                        )

            if now - last_seen >= cfg.hold_seconds:
                tracking_active = False
                stable_frames = 0
                locked_track_id = None
                if not released:
                    print("no target -> releasing", flush=True)
                    hardware.publish(
                        ArmCommand(
                            q=state.arm_q,
                            dq=tuple(0.0 for _ in state.arm_q),
                            kp=tuple(cfg.kp for _ in state.arm_q),
                            kd=tuple(cfg.kd for _ in state.arm_q),
                            weight=0.0,
                            mode_machine=int(state.mode_machine),
                            published_at=_now(),
                        )
                    )
                    released = True

            if tracking_active:
                # Publish continuously at the SDK's reference 50 Hz cadence;
                # only the desired pose is updated by fresh GB10 detections.
                target_q[0] += _clamp(desired_q[0] - target_q[0], -cfg.slew_step_rad, cfg.slew_step_rad)
                target_q[1] += _clamp(desired_q[1] - target_q[1], -cfg.slew_step_rad, cfg.slew_step_rad)
                target_q[2] += _clamp(desired_q[2] - target_q[2], -cfg.slew_step_rad, cfg.slew_step_rad)
                full_q = list(state.arm_q)
                full_q[7:14] = target_q
                hardware.publish(
                    _make_command(
                        state,
                        full_q[7],
                        full_q[8],
                        full_q[9],
                        kp_value=cfg.kp,
                        kd_value=cfg.kd,
                    )
                )

            elapsed = _now() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("stopping: releasing arm")
        state = hardware.latest_state()
        if state is not None:
            release = ArmCommand(
                q=state.arm_q,
                dq=tuple(0.0 for _ in state.arm_q),
                kp=tuple(60.0 for _ in state.arm_q),
                kd=tuple(1.5 for _ in state.arm_q),
                weight=0.0,
                mode_machine=int(state.mode_machine),
                published_at=_now(),
            )
            try:
                hardware.publish(release)
            except Exception:
                pass
        return 0
    finally:
        poller.stop()
        hardware.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
