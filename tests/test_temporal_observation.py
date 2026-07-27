import pytest

from object_tracking.temporal_observation import (
    TemporalObservationBuffer,
    TimedObservation,
    temporal_indices,
)


def test_temporal_indices_use_requested_time_span() -> None:
    timestamps = [index * 16_666_667 for index in range(40)]
    indices = temporal_indices(timestamps, frame_count=5, span_s=0.5)
    assert indices[-1] == 39
    assert timestamps[indices[-1]] - timestamps[indices[0]] == pytest.approx(
        500_000_000, abs=17_000_000
    )
    assert len(set(indices)) == 5


def test_temporal_buffer_rejects_unsynchronized_state() -> None:
    buffer = TemporalObservationBuffer[str](max_sync_error_ms=10)
    with pytest.raises(ValueError, match="synchronization"):
        buffer.append(TimedObservation("frame", (0.0,), 20_000_000, 0, 1))


def test_temporal_buffer_returns_causal_window() -> None:
    buffer = TemporalObservationBuffer[str](max_duration_s=1.0)
    for index in range(40):
        timestamp = index * 16_666_667
        buffer.append(TimedObservation(str(index), (float(index),), timestamp, timestamp, index))
    window = buffer.window(frame_count=5, span_s=0.5)
    assert window[-1].frame == "39"
    assert [item.sequence for item in window] == sorted(item.sequence for item in window)


def test_temporal_indices_fail_when_history_is_too_short() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        temporal_indices([0, 20_000_000], frame_count=2, span_s=0.5)
