"""Offline action-target alignment helpers for VLA training.

The real robot and Isaac recordings use command targets with different
command-to-observed-state lags.  Mixing those commands gives the action tensor
source-dependent semantics.  This module instead constructs a common target:
the achieved state a fixed number of frames after the observation.

It intentionally has no ROS, Unitree SDK, TensorFlow, or simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FUTURE_STATE_TARGET_V1 = "achieved_future_state_v1"


@dataclass(frozen=True)
class AlignedSequence:
    """Observation/target arrays with a shared prediction-time contract."""

    observations: np.ndarray
    targets: np.ndarray
    lookahead_frames: int

    @property
    def transitions(self) -> int:
        return int(len(self.observations))


def align_achieved_future_state(
    states: np.ndarray,
    lookahead_frames: int,
) -> AlignedSequence:
    """Pair state[t] with state[t + lookahead] without terminal padding.

    The returned arrays are views when NumPy can provide them.  The final
    ``lookahead_frames`` observations are excluded because their labels are not
    observable inside the episode.  This is important: clamping those labels
    to the terminal state would silently train artificial zero-motion targets.
    """

    values = np.asarray(states)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("states must be a non-empty [T,D] array")
    if lookahead_frames < 1:
        raise ValueError("lookahead_frames must be positive")
    if len(values) <= lookahead_frames:
        raise ValueError(
            "episode must contain more frames than the requested lookahead"
        )
    return AlignedSequence(
        observations=values[:-lookahead_frames],
        targets=values[lookahead_frames:],
        lookahead_frames=lookahead_frames,
    )


def alignment_metadata(lookahead_frames: int, fps: float = 30.0) -> dict:
    """Return serializable provenance for a future-state action contract."""

    if lookahead_frames < 1 or fps <= 0:
        raise ValueError("lookahead_frames and fps must be positive")
    return {
        "version": FUTURE_STATE_TARGET_V1,
        "target": "observed_achieved_state",
        "anchor": "observation_state_at_t",
        "lookahead_frames": int(lookahead_frames),
        "lookahead_seconds": float(lookahead_frames / fps),
        "terminal_policy": "drop_unobservable_targets",
        "source_independent": True,
    }
