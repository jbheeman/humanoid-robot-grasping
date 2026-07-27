"""Validation and group-aware splitting for xr_teleoperate bunny episodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import median
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RealEpisodeQCConfig:
    expected_hz: float = 30.0
    minimum_hz: float = 25.0
    maximum_hz: float = 35.0
    minimum_pre_contact_s: float = 0.40
    maximum_contact_frame_disagreement: int = 2


@dataclass(frozen=True)
class EpisodeAudit:
    episode_id: str
    path: str
    accepted: bool
    reasons: tuple[str, ...]
    frames: int
    duration_s: float
    contact_frame: int | None
    group_id: str | None
    condition_id: str | None
    session_id: str | None
    metadata: Mapping[str, object]
    fingerprint: str | None


def _metadata(payload: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    info = payload.get("info")
    if isinstance(info, Mapping):
        collection = info.get("collection")
        if isinstance(collection, Mapping):
            metadata.update(collection)
    companion = directory / "episode_metadata.json"
    if companion.is_file():
        value = json.loads(companion.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{companion}: expected a JSON object")
        metadata.update(value)
    return metadata


def _timestamps_ns(frames: list[Mapping[str, Any]]) -> list[int]:
    timestamps: list[int] = []
    for frame in frames:
        timing = frame.get("timing")
        if not isinstance(timing, Mapping):
            return []
        value = timing.get("monotonic_time_ns")
        if value is None:
            value = timing.get("wall_time_ns")
        try:
            timestamps.append(int(value))
        except (TypeError, ValueError):
            return []
    return timestamps


def _contact(frame: Mapping[str, Any]) -> bool:
    annotations = frame.get("annotations")
    if not isinstance(annotations, Mapping):
        return False
    contact = annotations.get("right_palm_contact")
    return bool(contact.get("value", False)) if isinstance(contact, Mapping) else False


def _right_qpos(frame: Mapping[str, Any], field: str) -> object:
    section = frame.get(field)
    if not isinstance(section, Mapping):
        return None
    right = section.get("right_arm")
    return right.get("qpos") if isinstance(right, Mapping) else None


def _condition(metadata: Mapping[str, object]) -> tuple[str | None, list[str]]:
    required = (
        "rabbit_path_angle_deg",
        "rabbit_speed_m_s",
        "right_arm_start",
        "camera_view",
        "collection_session_id",
    )
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        return None, [f"missing_metadata:{key}" for key in missing]
    angle = round(float(metadata["rabbit_path_angle_deg"]) / 15.0) * 15
    speed = round(float(metadata["rabbit_speed_m_s"]), 2)
    condition = (
        f"angle={angle:+03d}|speed={speed:.2f}|"
        f"arm={metadata['right_arm_start']}|camera={metadata['camera_view']}"
    )
    return condition, []


def audit_episode(
    directory: Path,
    config: RealEpisodeQCConfig | None = None,
) -> EpisodeAudit:
    cfg = config or RealEpisodeQCConfig()
    reasons: list[str] = []
    data_path = directory / "data.json"
    if not data_path.is_file():
        return EpisodeAudit(
            directory.name, str(directory), False, ("missing_data_json",), 0, 0.0,
            None, None, None, None, {}, None
        )
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    frames_value = payload.get("data")
    frames = frames_value if isinstance(frames_value, list) else []
    if len(frames) < 2 or not all(isinstance(frame, Mapping) for frame in frames):
        reasons.append("invalid_or_short_frame_sequence")
        frames = []
    typed_frames: list[Mapping[str, Any]] = list(frames)
    metadata = _metadata(payload, directory)

    timestamps = _timestamps_ns(typed_frames)
    duration_s = 0.0
    if len(timestamps) == len(typed_frames) and len(timestamps) >= 2:
        deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if any(delta <= 0 for delta in deltas):
            reasons.append("timestamps_not_strictly_increasing")
        else:
            duration_s = (timestamps[-1] - timestamps[0]) / 1e9
            observed_hz = 1e9 / median(deltas)
            if not cfg.minimum_hz <= observed_hz <= cfg.maximum_hz:
                reasons.append(f"sample_rate_out_of_range:{observed_hz:.2f}")
    else:
        reasons.append("missing_timestamps")

    missing_images = 0
    invalid_state_frames = 0
    invalid_action_frames = 0
    for frame in typed_frames:
        colors = frame.get("colors")
        relative = colors.get("color_0") if isinstance(colors, Mapping) else None
        if not relative or not (directory / str(relative)).is_file():
            missing_images += 1
        state = _right_qpos(frame, "states")
        action = _right_qpos(frame, "actions")
        if not isinstance(state, list) or len(state) != 7:
            invalid_state_frames += 1
        if not isinstance(action, list) or len(action) != 7:
            invalid_action_frames += 1
    if missing_images:
        reasons.append(f"missing_rgb_frames:{missing_images}")
    if invalid_state_frames:
        reasons.append(f"invalid_right_arm_state_frames:{invalid_state_frames}")
    if invalid_action_frames:
        reasons.append(f"invalid_right_arm_action_frames:{invalid_action_frames}")

    contacts = [index for index, frame in enumerate(typed_frames) if _contact(frame)]
    contact_frame = contacts[0] if contacts else None
    if contact_frame is None:
        reasons.append("no_operator_contact_annotation")
    elif timestamps and contact_frame < len(timestamps):
        pre_contact_s = (timestamps[contact_frame] - timestamps[0]) / 1e9
        if pre_contact_s < cfg.minimum_pre_contact_s:
            reasons.append(f"insufficient_pre_contact_video:{pre_contact_s:.3f}")

    if metadata.get("contact_reviewed") is not True:
        reasons.append("contact_not_visually_reviewed")
    try:
        visual_contact_frame = int(metadata["visual_contact_frame"])
    except (KeyError, TypeError, ValueError):
        visual_contact_frame = None
        reasons.append("missing_visual_contact_frame")
    if (
        contact_frame is not None
        and visual_contact_frame is not None
        and abs(contact_frame - visual_contact_frame)
        > cfg.maximum_contact_frame_disagreement
    ):
        reasons.append(
            f"contact_frame_disagreement:{contact_frame}:{visual_contact_frame}"
        )

    if metadata.get("moving_object") is not True:
        reasons.append("moving_object_not_confirmed")
    condition_id, condition_reasons = _condition(metadata)
    reasons.extend(condition_reasons)
    session = metadata.get("collection_session_id")
    session_id = str(session) if session not in (None, "") else None
    block = metadata.get("collection_block_id")
    group_id = str(block) if block not in (None, "") else session_id

    fingerprint = None
    if typed_frames:
        colors = typed_frames[contact_frame or 0].get("colors")
        relative = colors.get("color_0") if isinstance(colors, Mapping) else None
        image_path = directory / str(relative) if relative else None
        if image_path is not None and image_path.is_file():
            fingerprint = hashlib.sha256(image_path.read_bytes()).hexdigest()

    return EpisodeAudit(
        episode_id=directory.name,
        path=str(directory),
        accepted=not reasons,
        reasons=tuple(reasons),
        frames=len(typed_frames),
        duration_s=duration_s,
        contact_frame=contact_frame,
        group_id=group_id,
        condition_id=condition_id,
        session_id=session_id,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def reject_duplicate_contact_frames(
    audits: Iterable[EpisodeAudit],
) -> tuple[EpisodeAudit, ...]:
    seen: dict[str, str] = {}
    result: list[EpisodeAudit] = []
    for audit in audits:
        reasons = list(audit.reasons)
        if audit.fingerprint:
            duplicate = seen.get(audit.fingerprint)
            if duplicate is not None:
                reasons.append(f"duplicate_contact_frame:{duplicate}")
            else:
                seen[audit.fingerprint] = audit.episode_id
        result.append(
            EpisodeAudit(
                **{
                    **audit.__dict__,
                    "accepted": not reasons,
                    "reasons": tuple(reasons),
                }
            )
        )
    return tuple(result)


def grouped_split(
    audits: Iterable[EpisodeAudit],
    *,
    seed: int = 20260723,
) -> dict[str, str]:
    accepted = [audit for audit in audits if audit.accepted]
    if len(accepted) < 30:
        raise ValueError("at least 30 accepted episodes are required for train/val/test")
    groups: dict[str, list[EpisodeAudit]] = defaultdict(list)
    for audit in accepted:
        if audit.group_id is None:
            raise ValueError(f"{audit.episode_id}: missing group id")
        groups[audit.group_id].append(audit)
    if len(groups) < 3:
        raise ValueError("at least three independent collection groups are required")

    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    targets = {"train": 0.67 * len(accepted), "val": 0.165 * len(accepted), "test": 0.165 * len(accepted)}
    counts = Counter({"train": 0, "val": 0, "test": 0})
    assignments: dict[str, str] = {}
    for key in keys:
        split = min(
            targets,
            key=lambda name: (counts[name] / max(targets[name], 1.0), counts[name]),
        )
        for audit in groups[key]:
            assignments[audit.episode_id] = split
        counts[split] += len(groups[key])
    if any(counts[name] == 0 for name in targets):
        raise ValueError(f"grouped split produced an empty partition: {dict(counts)}")
    return assignments


def grouped_cross_validation_folds(
    audits: Iterable[EpisodeAudit],
    *,
    folds: int = 5,
) -> dict[str, int]:
    accepted = [audit for audit in audits if audit.accepted]
    groups = sorted({audit.group_id for audit in accepted if audit.group_id is not None})
    if len(accepted) < 20:
        raise ValueError("at least 20 accepted episodes are required for pilot CV")
    if len(groups) < 2:
        raise ValueError("at least two independent collection groups are required")
    fold_count = min(folds, len(groups))
    group_folds = {group: index % fold_count for index, group in enumerate(groups)}
    return {
        audit.episode_id: group_folds[audit.group_id]
        for audit in accepted
        if audit.group_id is not None
    }
