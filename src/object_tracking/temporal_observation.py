"""Timestamped temporal observations for latency-aware VLA inference.

The robot camera may publish at 60 Hz while the policy is trained/evaluated at
30 Hz.  This module keeps timestamps authoritative: frames are selected by
time span, never by assuming a fixed input frame rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


FrameT = TypeVar("FrameT")


@dataclass(frozen=True)
class TimedObservation(Generic[FrameT]):
    frame: FrameT
    proprio: tuple[float, ...]
    capture_time_ns: int
    state_time_ns: int
    sequence: int

    @property
    def synchronization_error_ms(self) -> float:
        return abs(self.capture_time_ns - self.state_time_ns) / 1_000_000.0


def temporal_indices(
    timestamps_ns: Sequence[int],
    *,
    frame_count: int,
    span_s: float,
) -> tuple[int, ...]:
    """Select causal, approximately uniform samples ending at the latest frame."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if span_s < 0:
        raise ValueError("span_s cannot be negative")
    if not timestamps_ns:
        raise ValueError("at least one timestamp is required")
    timestamps = tuple(int(value) for value in timestamps_ns)
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if frame_count == 1:
        return (len(timestamps) - 1,)

    latest = timestamps[-1]
    requested_start = latest - int(span_s * 1_000_000_000)
    if timestamps[0] > requested_start:
        raise ValueError("buffer does not cover the requested temporal span")

    targets = [
        requested_start + round(index * (latest - requested_start) / (frame_count - 1))
        for index in range(frame_count)
    ]
    selected: list[int] = []
    lower_bound = 0
    for target in targets:
        candidates = range(lower_bound, len(timestamps))
        chosen = min(candidates, key=lambda index: (abs(timestamps[index] - target), index))
        selected.append(chosen)
        lower_bound = chosen + 1
        if lower_bound > len(timestamps) and len(selected) < frame_count:
            raise ValueError("not enough distinct frames for requested window")
    if len(set(selected)) != frame_count:
        raise ValueError("not enough distinct frames for requested window")
    return tuple(selected)


class TemporalObservationBuffer(Generic[FrameT]):
    def __init__(self, *, max_duration_s: float = 1.5, max_sync_error_ms: float = 20.0) -> None:
        if max_duration_s <= 0 or max_sync_error_ms < 0:
            raise ValueError("invalid temporal buffer limits")
        self.max_duration_ns = int(max_duration_s * 1_000_000_000)
        self.max_sync_error_ms = float(max_sync_error_ms)
        self._observations: deque[TimedObservation[FrameT]] = deque()

    def append(self, observation: TimedObservation[FrameT]) -> None:
        if self._observations:
            latest = self._observations[-1]
            if observation.capture_time_ns <= latest.capture_time_ns:
                raise ValueError("capture timestamps must increase")
            if observation.sequence <= latest.sequence:
                raise ValueError("frame sequence must increase")
        if observation.synchronization_error_ms > self.max_sync_error_ms:
            raise ValueError(
                f"image/proprio synchronization error is too large "
                f"({observation.synchronization_error_ms:.1f} ms)"
            )
        self._observations.append(observation)
        cutoff = observation.capture_time_ns - self.max_duration_ns
        while len(self._observations) > 1 and self._observations[0].capture_time_ns < cutoff:
            self._observations.popleft()

    def window(self, *, frame_count: int, span_s: float) -> tuple[TimedObservation[FrameT], ...]:
        observations = tuple(self._observations)
        indices = temporal_indices(
            [item.capture_time_ns for item in observations],
            frame_count=frame_count,
            span_s=span_s,
        )
        return tuple(observations[index] for index in indices)

    def __len__(self) -> int:
        return len(self._observations)
