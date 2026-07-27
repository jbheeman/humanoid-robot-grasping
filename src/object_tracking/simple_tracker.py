from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    first_seen: float
    last_seen: float
    velocity_px_per_sec: list[float] = field(default_factory=lambda: [0.0, 0.0])
    age_frames: int = 1
    missed_updates: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "center_xy": self.center_xy,
            "velocity_px_per_sec": self.velocity_px_per_sec,
            "age_frames": self.age_frames,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "missed_updates": self.missed_updates,
        }


class SimpleTracker:
    """Small center-distance tracker for low-speed indoor object demos."""

    def __init__(
        self,
        max_center_distance_px: float = 90.0,
        max_missed_updates: int = 10,
    ) -> None:
        self.max_center_distance_px = max_center_distance_px
        self.max_missed_updates = max_missed_updates
        self._next_track_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[dict[str, Any]], timestamp: float) -> list[dict[str, Any]]:
        unmatched_track_ids = set(self._tracks)

        for detection in detections:
            center = [float(detection["center_xy"][0]), float(detection["center_xy"][1])]
            class_name = str(detection["class_name"])
            best_track_id = self._match_track(class_name, center, unmatched_track_ids)

            if best_track_id is None:
                track = self._new_track(detection, timestamp)
                self._tracks[track.track_id] = track
                continue

            track = self._tracks[best_track_id]
            unmatched_track_ids.discard(best_track_id)
            dt = max(timestamp - track.last_seen, 1e-6)
            vx = (center[0] - track.center_xy[0]) / dt
            vy = (center[1] - track.center_xy[1]) / dt

            track.class_name = class_name
            track.confidence = float(detection["confidence"])
            track.bbox_xyxy = [float(v) for v in detection["bbox_xyxy"]]
            track.center_xy = center
            track.velocity_px_per_sec = [vx, vy]
            track.last_seen = timestamp
            track.age_frames += 1
            track.missed_updates = 0

        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missed_updates += 1
            if track.missed_updates > self.max_missed_updates:
                del self._tracks[track_id]

        return self.tracks()

    def tracks(self) -> list[dict[str, Any]]:
        return [track.as_dict() for track in sorted(self._tracks.values(), key=lambda item: item.track_id)]

    def _match_track(
        self,
        class_name: str,
        center: list[float],
        candidate_track_ids: set[int],
    ) -> int | None:
        best_track_id = None
        best_distance = self.max_center_distance_px

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            if track.class_name != class_name:
                continue

            distance = hypot(center[0] - track.center_xy[0], center[1] - track.center_xy[1])
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        return best_track_id

    def _new_track(self, detection: dict[str, Any], timestamp: float) -> Track:
        track_id = self._next_track_id
        self._next_track_id += 1
        return Track(
            track_id=track_id,
            class_name=str(detection["class_name"]),
            confidence=float(detection["confidence"]),
            bbox_xyxy=[float(v) for v in detection["bbox_xyxy"]],
            center_xy=[float(detection["center_xy"][0]), float(detection["center_xy"][1])],
            first_seen=timestamp,
            last_seen=timestamp,
        )
