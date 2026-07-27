from __future__ import annotations

import numpy as np
import pytest

from object_tracking.trajectory_forecaster import trajectory_features
from object_tracking.synthetic_trajectory import synthetic_batch


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


def test_synthetic_batch_is_finite_and_reproducible() -> None:
    first = synthetic_batch(np.random.default_rng(7), batch_size=8, history=12, horizons_s=(0.1, 0.15, 0.25))
    second = synthetic_batch(np.random.default_rng(7), batch_size=8, history=12, horizons_s=(0.1, 0.15, 0.25))
    assert first[0].shape == (8, 12, 6)
    assert first[1].shape == (8, 3, 3)
    assert np.all(np.isfinite(first[0]))
    assert np.all(np.isfinite(first[1]))
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
