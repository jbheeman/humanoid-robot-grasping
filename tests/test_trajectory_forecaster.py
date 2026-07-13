from __future__ import annotations

import numpy as np
import pytest

from object_tracking.trajectory_forecaster import trajectory_features


def test_trajectory_features_are_translation_invariant() -> None:
    times = [1.0, 1.1, 1.2]
    positions = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
    moved = [[x + 4.0, y - 3.0, z + 2.0] for x, y, z in positions]
    np.testing.assert_allclose(
        trajectory_features(times, positions), trajectory_features(times, moved), atol=2e-6
    )


def test_trajectory_features_encode_velocity() -> None:
    features = trajectory_features([0.0, 0.1, 0.2], [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    np.testing.assert_allclose(features[:, 3:], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(features[-1, :3], [0, 0, 0])


def test_trajectory_features_reject_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_features([0.0, 0.0], [[0, 0, 0], [1, 0, 0]])
