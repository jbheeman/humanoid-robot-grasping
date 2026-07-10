"""Depth-frame pairing and robust object-depth estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    sequence: int
    receipt_time_s: float
    z16: np.ndarray
    depth_scale: float
    calibration_id: str
    sensor_timestamp_ms: float | None = None
    registered_to_rgb: bool = False

    def __post_init__(self) -> None:
        array = np.asarray(self.z16)
        if array.ndim != 2 or array.dtype.kind != "u" or array.dtype.itemsize != 2:
            raise ValueError("z16 must be a two-dimensional uint16 array")
        if self.sequence < 0 or not np.isfinite(self.receipt_time_s) or self.receipt_time_s < 0:
            raise ValueError("sequence and receipt time must be finite and non-negative")
        if not 0.0 < self.depth_scale < 1.0:
            raise ValueError("depth_scale must be between zero and one")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        object.__setattr__(self, "z16", array)


@dataclass(frozen=True)
class DepthPair:
    rgb_receipt_time_s: float
    depth: DepthFrame
    skew_s: float


@dataclass(frozen=True)
class DepthEstimate:
    depth_m: float
    pixel_xy: tuple[float, float]
    sample_count: int
    valid_fraction: float
    mad_m: float
    cluster_span_m: float

    @property
    def is_certain(self) -> bool:
        return self.sample_count >= 8 and self.mad_m <= 0.03


class DepthFrameBuffer:
    """Small newest-frame buffer keyed only by GB10 monotonic receipt time."""

    def __init__(self, maxlen: int = 8) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._frames: deque[DepthFrame] = deque(maxlen=maxlen)

    def add(self, frame: DepthFrame) -> None:
        if self._frames and frame.sequence <= self._frames[-1].sequence:
            raise ValueError("depth sequence must be strictly increasing")
        self._frames.append(frame)

    def newest(self) -> DepthFrame | None:
        return self._frames[-1] if self._frames else None

    def pair(self, rgb_receipt_time_s: float, max_skew_s: float = 0.100) -> DepthPair | None:
        return pair_rgb_depth(rgb_receipt_time_s, self._frames, max_skew_s=max_skew_s)


def pair_rgb_depth(
    rgb_receipt_time_s: float,
    depth_frames: Iterable[DepthFrame],
    *,
    max_skew_s: float = 0.100,
) -> DepthPair | None:
    """Pair by a single monotonic receipt clock, never by sensor clocks."""

    if (
        not np.isfinite(rgb_receipt_time_s)
        or not np.isfinite(max_skew_s)
        or rgb_receipt_time_s < 0
        or max_skew_s < 0
    ):
        raise ValueError("timestamps and max_skew_s must be finite and non-negative")
    nearest: DepthFrame | None = None
    nearest_skew = float("inf")
    for frame in depth_frames:
        skew = abs(frame.receipt_time_s - rgb_receipt_time_s)
        if skew < nearest_skew or (
            skew == nearest_skew and nearest is not None and frame.sequence > nearest.sequence
        ):
            nearest, nearest_skew = frame, skew
    if nearest is None or nearest_skew > max_skew_s:
        return None
    return DepthPair(rgb_receipt_time_s, nearest, nearest_skew)


def central_roi(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
    *,
    fraction: float = 0.6,
) -> tuple[int, int, int, int]:
    """Return a clipped integer central ROI as ``x1, y1, x2, y2`` (exclusive)."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    height, width = image_shape
    x1, y1, x2, y2 = (float(item) for item in bbox_xyxy)
    if not np.all(np.isfinite((x1, y1, x2, y2))):
        raise ValueError("bbox values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = (x2 - x1) * fraction / 2.0
    half_height = (y2 - y1) * fraction / 2.0
    left = max(0, min(width, int(np.floor(center_x - half_width))))
    top = max(0, min(height, int(np.floor(center_y - half_height))))
    right = max(0, min(width, int(np.ceil(center_x + half_width))))
    bottom = max(0, min(height, int(np.ceil(center_y + half_height))))
    if right <= left or bottom <= top:
        raise ValueError("bbox does not overlap the image")
    return left, top, right, bottom


def estimate_roi_depth(
    z16: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    *,
    depth_scale: float,
    roi_fraction: float = 0.6,
    min_depth_m: float = 0.12,
    max_depth_m: float = 4.0,
    min_samples: int = 8,
    min_valid_fraction: float = 0.05,
    cluster_gap_m: float = 0.04,
) -> DepthEstimate | None:
    """Estimate foreground depth from the strongest valid central-ROI cluster.

    Contiguous sorted-depth clusters are split at a metric gap.  Clusters with
    adequate support are ranked by sample count and then proximity, preventing a
    few foreground outliers from beating the object surface while preferring the
    near surface when support is tied.
    """

    array = np.asarray(z16)
    if array.ndim != 2:
        raise ValueError("z16 must be two-dimensional")
    if (
        not np.isfinite(depth_scale)
        or depth_scale <= 0
        or min_samples <= 0
        or cluster_gap_m <= 0
        or not 0 <= min_valid_fraction <= 1
        or min_depth_m >= max_depth_m
    ):
        raise ValueError("depth_scale, min_samples, and cluster_gap_m must be positive")
    left, top, right, bottom = central_roi(bbox_xyxy, array.shape, fraction=roi_fraction)
    roi = array[top:bottom, left:right].astype(np.float64) * depth_scale
    valid = np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)
    valid_count = int(valid.sum())
    total_count = int(roi.size)
    valid_fraction = valid_count / total_count if total_count else 0.0
    if valid_count < min_samples or valid_fraction < min_valid_fraction:
        return None

    ys, xs = np.nonzero(valid)
    values = roi[valid]
    order = np.argsort(values)
    sorted_values = values[order]
    split_indices = np.flatnonzero(np.diff(sorted_values) > cluster_gap_m) + 1
    clusters = np.split(order, split_indices)
    eligible = [indices for indices in clusters if len(indices) >= min_samples]
    if not eligible:
        return None
    max_support = max(len(indices) for indices in eligible)
    supported = [indices for indices in eligible if len(indices) >= max_support * 0.6]
    chosen = min(supported, key=lambda indices: float(np.median(values[indices])))
    chosen_values = values[chosen]
    median = float(np.median(chosen_values))
    deviations = np.abs(chosen_values - median)
    mad = float(np.median(deviations))
    inlier_tolerance = max(0.01, 3.0 * mad)
    representative = chosen[deviations <= inlier_tolerance]
    pixel_x = float(left + np.median(xs[representative]))
    pixel_y = float(top + np.median(ys[representative]))
    return DepthEstimate(
        depth_m=median,
        pixel_xy=(pixel_x, pixel_y),
        sample_count=len(chosen),
        valid_fraction=valid_fraction,
        mad_m=mad,
        cluster_span_m=float(np.ptp(chosen_values)),
    )
