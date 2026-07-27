"""Fail-closed scheduler for asynchronous VLA action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import bisect
import math
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
    max_state_observation_skew_s: float = 0.025
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
    waypoint_index: int | None = None
    waypoint_time_s: float | None = None
    expired_waypoints: int = 0


@dataclass(frozen=True)
class ChunkTiming:
    """Clock contract for one asynchronous policy proposal.

    Waypoint zero is the first 30 Hz target *after* the state anchor, rather
    than a target to replay when inference eventually completes.
    """

    observation_time_s: float
    state_anchor_time_s: float
    inference_started_time_s: float
    inference_completed_time_s: float
    submitted_time_s: float


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
        self._timing: ChunkTiming | None = None
        self._waypoint_times_s: tuple[float, ...] = ()
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
        state_anchor_time_s: float | None = None,
        inference_started_time_s: float | None = None,
    ) -> ScheduledAction:
        values = np.asarray(actions, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.action_dimension or not len(values):
            return ScheduledAction(ScheduleDecision.REJECT, None, "invalid_action_shape")
        if not np.all(np.isfinite(values)):
            return ScheduledAction(ScheduleDecision.REJECT, None, "nonfinite_action")
        anchor_time = observation_time_s if state_anchor_time_s is None else state_anchor_time_s
        inference_started = (
            observation_time_s
            if inference_started_time_s is None
            else inference_started_time_s
        )
        timestamps = (
            observation_time_s,
            anchor_time,
            inference_started,
            inference_completed_time_s,
            now_s,
        )
        if not all(math.isfinite(value) for value in timestamps):
            return ScheduledAction(ScheduleDecision.REJECT, None, "invalid_timestamps")
        if observation_time_s <= self._last_source_time_s:
            return ScheduledAction(ScheduleDecision.REJECT, None, "out_of_order_observation")
        if (
            inference_started < observation_time_s
            or inference_completed_time_s < inference_started
            or now_s < inference_completed_time_s
        ):
            return ScheduledAction(ScheduleDecision.REJECT, None, "invalid_timestamps")
        if abs(anchor_time - observation_time_s) > self.config.max_state_observation_skew_s:
            return ScheduledAction(
                ScheduleDecision.REJECT, None, "state_observation_desynchronized"
            )
        observation_age = now_s - observation_time_s
        inference_latency = inference_completed_time_s - inference_started
        if observation_age > self.config.max_observation_age_s:
            return ScheduledAction(
                ScheduleDecision.REJECT, None, "stale_observation", observation_age
            )
        if inference_latency > self.config.max_inference_latency_s:
            return ScheduledAction(
                ScheduleDecision.REJECT, None, "late_inference", observation_age
            )

        waypoint_times = tuple(
            anchor_time + (index + 1) * self.config.control_period_s
            for index in range(len(values))
        )
        cursor = bisect.bisect_left(waypoint_times, now_s)
        if cursor >= len(values):
            return ScheduledAction(
                ScheduleDecision.REJECT,
                None,
                "chunk_fully_expired",
                observation_age,
                expired_waypoints=cursor,
            )

        values = values.copy()
        if self._last_action is not None and self.config.blend_steps > 0:
            count = min(self.config.blend_steps, len(values) - cursor)
            for blend_index in range(count):
                index = cursor + blend_index
                alpha = (blend_index + 1) / (count + 1)
                values[index] = (
                    (1.0 - alpha) * self._last_action + alpha * values[index]
                )

        self._chunk = values
        self._cursor = cursor
        self._source_time_s = float(observation_time_s)
        self._timing = ChunkTiming(
            observation_time_s=float(observation_time_s),
            state_anchor_time_s=float(anchor_time),
            inference_started_time_s=float(inference_started),
            inference_completed_time_s=float(inference_completed_time_s),
            submitted_time_s=float(now_s),
        )
        self._waypoint_times_s = waypoint_times
        self._last_source_time_s = float(observation_time_s)
        self._cancel_reason = None
        return ScheduledAction(
            ScheduleDecision.EXECUTE,
            None,
            "chunk_accepted",
            observation_age,
            waypoint_index=cursor,
            waypoint_time_s=waypoint_times[cursor],
            expired_waypoints=cursor,
        )

    def cancel(self, reason: str) -> ScheduledAction:
        self._chunk = None
        self._cursor = 0
        self._timing = None
        self._waypoint_times_s = ()
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
            self.cancel("chunk_expired")
            return ScheduledAction(
                ScheduleDecision.HOLD, None, "chunk_expired", observation_age
            )
        expired_cursor = bisect.bisect_left(self._waypoint_times_s, now_s)
        skipped = max(0, expired_cursor - self._cursor)
        self._cursor = max(self._cursor, expired_cursor)
        if self._cursor >= len(self._chunk):
            self.cancel("no_unexpired_waypoints")
            return ScheduledAction(
                ScheduleDecision.HOLD,
                None,
                "no_unexpired_waypoints",
                observation_age,
                expired_waypoints=skipped,
            )

        waypoint_index = self._cursor
        waypoint_time_s = self._waypoint_times_s[waypoint_index]
        action = self._chunk[self._cursor].copy()
        if self._last_action is not None:
            maximum = np.broadcast_to(
                np.asarray(self.config.maximum_step, dtype=float), (self.action_dimension,)
            )
            delta = np.clip(action - self._last_action, -maximum, maximum)
            action = self._last_action + delta
        self._cursor += 1
        self._last_action = action
        return ScheduledAction(
            ScheduleDecision.EXECUTE,
            action,
            "scheduled",
            observation_age,
            waypoint_index=waypoint_index,
            waypoint_time_s=waypoint_time_s,
            expired_waypoints=skipped,
        )
