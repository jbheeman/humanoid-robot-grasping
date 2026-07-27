"""Task-aware QC policy for moving-plush xr_teleoperate demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RealVisualQCConfig:
    action_hz: float = 30.0
    minimum_pre_contact_s: float = 0.40
    maximum_episode_s: float = 30.0
    minimum_visible_fraction: float = 0.60
    minimum_path_displacement_px: float = 30.0
    post_contact_trim_frames: int = 9
    maximum_contact_transitions: int = 4


def contact_runs(contact: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(contact):
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(contact)))
    return tuple(runs)


def structural_reasons(
    metrics: Mapping[str, object],
    config: RealVisualQCConfig | None = None,
) -> tuple[str, ...]:
    cfg = config or RealVisualQCConfig()
    reasons: list[str] = []
    frames = int(metrics["frames"])
    duration_s = float(metrics["duration_s"])
    hz = float(metrics["sample_hz"])
    first_contact = int(metrics["first_contact_frame"])
    if frames < int(math.ceil(cfg.minimum_pre_contact_s * cfg.action_hz)) + 2:
        reasons.append("episode_too_short")
    if not 25.0 <= hz <= 35.0:
        reasons.append("sample_rate_out_of_range")
    if int(metrics["missing_images"]) > 0:
        reasons.append("missing_rgb_frames")
    if int(metrics["invalid_joint_frames"]) > 0:
        reasons.append("invalid_right_arm_vectors")
    if first_contact < 0:
        reasons.append("no_contact_annotation")
    elif first_contact / cfg.action_hz < cfg.minimum_pre_contact_s:
        reasons.append("contact_starts_too_early")
    if duration_s > cfg.maximum_episode_s:
        reasons.append("episode_too_long")
    if int(metrics["contact_transitions"]) > cfg.maximum_contact_transitions:
        reasons.append("contact_annotation_chatter")
    return tuple(reasons)


def maximum_center_displacement_px(
    centers: Sequence[tuple[float, float] | None],
) -> float:
    visible = [center for center in centers if center is not None]
    maximum = 0.0
    for index, left in enumerate(visible):
        for right in visible[index + 1 :]:
            maximum = max(maximum, math.dist(left, right))
    return maximum


def visual_reasons(
    *,
    centers: Sequence[tuple[float, float] | None],
    contact_visible: bool,
    config: RealVisualQCConfig | None = None,
) -> tuple[str, ...]:
    cfg = config or RealVisualQCConfig()
    reasons: list[str] = []
    visible_fraction = (
        sum(center is not None for center in centers) / len(centers)
        if centers
        else 0.0
    )
    if visible_fraction < cfg.minimum_visible_fraction:
        reasons.append("plush_visibility_low")
    if not contact_visible:
        reasons.append("plush_not_visible_at_contact")
    if maximum_center_displacement_px(centers) < cfg.minimum_path_displacement_px:
        reasons.append("plush_motion_too_small")
    return tuple(reasons)


def training_frame_range(
    frame_count: int,
    first_contact_frame: int,
    config: RealVisualQCConfig | None = None,
) -> tuple[int, int]:
    """Return [start,end) while excluding human reset activity after contact."""

    cfg = config or RealVisualQCConfig()
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 0 <= first_contact_frame < frame_count:
        raise ValueError("first contact frame is outside the episode")
    return 0, min(frame_count, first_contact_frame + cfg.post_contact_trim_frames + 1)


def classify_episode(
    *,
    operator_rejected: bool,
    structural: Sequence[str],
    visual: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    if operator_rejected:
        return "operator_reject", ("operator_rejected",)
    if structural:
        return "automatic_reject", tuple(structural)
    if visual:
        return "needs_review", tuple(visual)
    return "accepted", ()
