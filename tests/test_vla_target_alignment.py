from __future__ import annotations

import numpy as np
import pytest

from object_tracking.vla_target_alignment import (
    FUTURE_STATE_TARGET_V1,
    align_achieved_future_state,
    alignment_metadata,
)


def test_future_state_alignment_has_exact_index_semantics() -> None:
    states = np.arange(30, dtype=np.float32).reshape(10, 3)

    aligned = align_achieved_future_state(states, lookahead_frames=3)

    np.testing.assert_array_equal(aligned.observations, states[:7])
    np.testing.assert_array_equal(aligned.targets, states[3:])
    assert aligned.transitions == 7
    assert aligned.lookahead_frames == 3


def test_future_state_alignment_drops_unobservable_terminal_labels() -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)

    aligned = align_achieved_future_state(states, lookahead_frames=1)

    assert aligned.observations.shape == aligned.targets.shape == (3, 3)
    assert not np.array_equal(aligned.targets[-1], aligned.observations[-1])


@pytest.mark.parametrize("lookahead", (0, -1, 4))
def test_future_state_alignment_rejects_invalid_lookahead(lookahead: int) -> None:
    with pytest.raises(ValueError):
        align_achieved_future_state(np.zeros((4, 3), dtype=np.float32), lookahead)


def test_alignment_metadata_is_explicit_and_source_independent() -> None:
    metadata = alignment_metadata(3)

    assert metadata["version"] == FUTURE_STATE_TARGET_V1
    assert metadata["lookahead_seconds"] == pytest.approx(0.1)
    assert metadata["terminal_policy"] == "drop_unobservable_targets"
    assert metadata["source_independent"] is True
