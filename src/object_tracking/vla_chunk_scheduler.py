"""Fail-closed scheduler for asynchronous VLA action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class ScheduleDecision(str, Enum):
    EXECUTE = "execute"
    HOLD = "hold"
    CANCEL = "cancel"
    REJECT = "reject"


@dataclass(frozen=True)
class SchedulerConfig:
    control_period_s: float = 1.0 / 30.0
    # The measured GB10 policy latency is about 0.37--0.44 s and simulation
    # trains with 0.50--0.60 s delay. Reject anything outside that envelope.
    max_observation_age_s: float = 0.65
    max_inference_latency_s: float = 0.60
    minimum_tracker_confidence: float = 0.45
    minimum_table_clearance_m: float = 0.05
    blend_steps: int = 3
    maximum_step: float | Sequence[float] = 0.025


@dataclass(frozen=True)
class SafetySignal:
    tracker_confidence: float = 1.0
    tracker_veto: bool = False
    contact: bool = False
    table_clearance_m: float = float("inf")


@dataclass(frozen=True)
class ScheduledAction:
    decision: ScheduleDecision
    action: np.ndarray | None
    reason: str
    observation_age_s: float | None = None


class VLAChunkScheduler:
    """Accept fresh chunks and emit one bounded action per control tick.

    A new chunk replaces the unexecuted tail of an old chunk.  Its first few
    commands are blended from the last emitted command, avoiding a discontinuity
    while still letting fresh observations supersede stale predictions.
    """

    def __init__(self, action_dimension: int, config: SchedulerConfig | None = None) -> None:
        if action_dimension < 1:
            raise ValueError("action_dimension must be positive")
        self.action_dimension = action_dimension
        self.config = config or SchedulerConfig()
        self._chunk: np.ndarray | None = None
        self._cursor = 0
        self._source_time_s: float | None = None
        self._accepted_at_s: float | None = None
        self._last_source_time_s = float("-inf")
        self._last_action: np.ndarray | None = None
        self._cancel_reason: str | None = None

    def submit(
        self,
        actions: Sequence[Sequence[float]] | np.ndarray,
        *,
        observation_time_s: float,
        inference_completed_time_s: float,
        now_s: float,
    ) -> ScheduledAction:
        values = np.asarray(actions, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.action_dimension or not len(values):
            return ScheduledAction(ScheduleDecision.REJECT, None, "invalid_action_shape")
        if not np.all(np.isfinite(values)):
            return ScheduledAction(ScheduleDecision.REJECT, None, "nonfinite_action")
        if observation_time_s <= self._last_source_time_s:
            return ScheduledAction(ScheduleDecision.REJECT, None, "out_of_order_observation")
        if inference_completed_time_s < observation_time_s or now_s < inference_completed_time_s:
            return ScheduledAction(ScheduleDecision.REJECT, None, "invalid_timestamps")
        observation_age = now_s - observation_time_s
        inference_latency = inference_completed_time_s - observation_time_s
        if observation_age > self.config.max_observation_age_s:
            return ScheduledAction(
                ScheduleDecision.REJECT, None, "stale_observation", observation_age
            )
        if inference_latency > self.config.max_inference_latency_s:
            return ScheduledAction(
                ScheduleDecision.REJECT, None, "late_inference", observation_age
            )

        values = values.copy()
        if self._last_action is not None and self.config.blend_steps > 0:
            count = min(self.config.blend_steps, len(values))
            for index in range(count):
                alpha = (index + 1) / (count + 1)
                values[index] = (1.0 - alpha) * self._last_action + alpha * values[index]

        self._chunk = values
        self._cursor = 0
        self._source_time_s = float(observation_time_s)
        self._accepted_at_s = float(now_s)
        self._last_source_time_s = float(observation_time_s)
        self._cancel_reason = None
        return ScheduledAction(ScheduleDecision.EXECUTE, None, "chunk_accepted", observation_age)

    def cancel(self, reason: str) -> ScheduledAction:
        self._chunk = None
        self._cursor = 0
        self._cancel_reason = reason
        return ScheduledAction(ScheduleDecision.CANCEL, None, reason)

    def tick(self, *, now_s: float, safety: SafetySignal) -> ScheduledAction:
        if safety.contact:
            return self.cancel("contact_detected")
        if safety.tracker_veto:
            return self.cancel("tracker_veto")
        if safety.tracker_confidence < self.config.minimum_tracker_confidence:
            return self.cancel("tracker_confidence_low")
        if safety.table_clearance_m < self.config.minimum_table_clearance_m:
            return self.cancel("table_clearance_low")
        if self._chunk is None or self._source_time_s is None:
            reason = self._cancel_reason or "no_fresh_chunk"
            return ScheduledAction(ScheduleDecision.HOLD, None, reason)
        observation_age = now_s - self._source_time_s
        if observation_age > self.config.max_observation_age_s:
            self._chunk = None
            return ScheduledAction(
                ScheduleDecision.HOLD, None, "chunk_expired", observation_age
            )
        if self._cursor >= len(self._chunk):
            self._chunk = None
            return ScheduledAction(ScheduleDecision.HOLD, None, "chunk_exhausted", observation_age)

        action = self._chunk[self._cursor].copy()
        if self._last_action is not None:
            maximum = np.broadcast_to(
                np.asarray(self.config.maximum_step, dtype=float), (self.action_dimension,)
            )
            delta = np.clip(action - self._last_action, -maximum, maximum)
            action = self._last_action + delta
        self._cursor += 1
        self._last_action = action
        return ScheduledAction(ScheduleDecision.EXECUTE, action, "scheduled", observation_age)
