"""Independent detector/tracker safety reference for VLA execution.

Detector output is deliberately not part of the VLA observation.  It can only
veto/hold a proposed action, so detector mistakes cannot steer the arm toward a
different target.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Sequence

from .vla_chunk_scheduler import SafetySignal


@dataclass(frozen=True)
class VisualSafetyConfig:
    target_classes: tuple[str, ...] = ("rabbit_plush", "bunny", "plush_rabbit")
    minimum_confidence: float = 0.45
    minimum_track_age_frames: int = 3
    maximum_track_age_s: float = 0.15
    maximum_missed_updates: int = 1
    maximum_image_speed_px_s: float = 1_500.0
    ambiguity_confidence_margin: float = 0.08


class VisualSafetyGovernor:
    def __init__(self, config: VisualSafetyConfig | None = None) -> None:
        self.config = config or VisualSafetyConfig()

    def signal(
        self,
        tracks: Sequence[dict[str, Any]],
        *,
        now_s: float,
        table_clearance_m: float,
        contact: bool = False,
    ) -> SafetySignal:
        candidates = [
            track for track in tracks
            if str(track.get("class_name", "")).lower() in self.config.target_classes
        ]
        candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        if not candidates:
            return SafetySignal(0.0, True, contact, table_clearance_m)
        best = candidates[0]
        confidence = float(best.get("confidence", 0.0))
        veto = confidence < self.config.minimum_confidence
        if len(candidates) > 1:
            runner_up = float(candidates[1].get("confidence", 0.0))
            veto |= confidence - runner_up < self.config.ambiguity_confidence_margin
        last_seen = float(best.get("last_seen", float("-inf")))
        veto |= now_s < last_seen or now_s - last_seen > self.config.maximum_track_age_s
        veto |= int(best.get("age_frames", 0)) < self.config.minimum_track_age_frames
        veto |= int(best.get("missed_updates", 0)) > self.config.maximum_missed_updates
        velocity = best.get("velocity_px_per_sec", (float("inf"), float("inf")))
        try:
            speed = hypot(float(velocity[0]), float(velocity[1]))
        except (IndexError, TypeError, ValueError):
            speed = float("inf")
        veto |= speed > self.config.maximum_image_speed_px_s
        return SafetySignal(confidence, veto, contact, table_clearance_m)
