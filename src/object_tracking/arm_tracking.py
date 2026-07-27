from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True)
class AimConfig:
    frame_width: int = 1280
    frame_height: int = 720
    prediction_s: float = 0.15
    deadzone: float = 0.05


@dataclass(frozen=True)
class AimCommand:
    track_id: int
    class_name: str
    confidence: float
    predicted_center_xy: tuple[float, float]
    x_error_norm: float
    y_error_norm: float
    source_timestamp: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "predicted_center_xy": list(self.predicted_center_xy),
            "x_error_norm": self.x_error_norm,
            "y_error_norm": self.y_error_norm,
            "source_timestamp": self.source_timestamp,
        }


def _bbox_area(track: dict[str, Any]) -> float:
    bbox = track.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) != 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(
        float(bbox[3]) - float(bbox[1]), 0.0
    )


def select_target(
    tracks: Iterable[dict[str, Any]],
    target_class: str = "plushie",
    min_confidence: float = 0.45,
    min_age_frames: int = 3,
    preferred_track_id: int | None = None,
) -> dict[str, Any] | None:
    candidates = [
        track
        for track in tracks
        if str(track.get("class_name", "")) == target_class
        and float(track.get("confidence", 0.0)) >= min_confidence
        and int(track.get("age_frames", 0)) >= min_age_frames
        and int(track.get("missed_updates", 0)) == 0
    ]
    if preferred_track_id is not None:
        for track in candidates:
            if int(track.get("track_id", -1)) == preferred_track_id:
                return track
    if not candidates:
        return None
    return max(candidates, key=lambda track: (float(track["confidence"]), _bbox_area(track)))


def _normalized_axis_error(value: float, center: float, half_extent: float, deadzone: float) -> float:
    raw = clamp((value - center) / max(half_extent, 1.0), -1.0, 1.0)
    if abs(raw) <= deadzone:
        return 0.0
    scaled = (abs(raw) - deadzone) / max(1.0 - deadzone, 1e-6)
    return scaled if raw > 0 else -scaled


def aim_from_track(track: dict[str, Any], config: AimConfig) -> AimCommand:
    center = track.get("center_xy") or [0.0, 0.0]
    velocity = track.get("velocity_px_per_sec") or [0.0, 0.0]
    predicted_x = clamp(
        float(center[0]) + float(velocity[0]) * config.prediction_s,
        0.0,
        float(config.frame_width - 1),
    )
    predicted_y = clamp(
        float(center[1]) + float(velocity[1]) * config.prediction_s,
        0.0,
        float(config.frame_height - 1),
    )
    x_error = _normalized_axis_error(
        predicted_x,
        config.frame_width / 2.0,
        config.frame_width / 2.0,
        config.deadzone,
    )
    y_error = _normalized_axis_error(
        predicted_y,
        config.frame_height / 2.0,
        config.frame_height / 2.0,
        config.deadzone,
    )
    return AimCommand(
        track_id=int(track["track_id"]),
        class_name=str(track["class_name"]),
        confidence=float(track["confidence"]),
        predicted_center_xy=(predicted_x, predicted_y),
        x_error_norm=x_error,
        y_error_norm=y_error,
        source_timestamp=float(track.get("last_seen", 0.0)),
    )
