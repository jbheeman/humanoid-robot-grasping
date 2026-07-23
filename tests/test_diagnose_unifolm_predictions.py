import numpy as np
import pytest

from scripts.training.diagnose_unifolm_predictions import latest_right_xyz


def test_latest_right_xyz_accepts_single_observation() -> None:
    state = np.arange(46, dtype=np.float32).reshape(2, 23)

    np.testing.assert_array_equal(latest_right_xyz(state), state[:, 9:12])


def test_latest_right_xyz_uses_newest_temporal_observation() -> None:
    state = np.arange(2 * 5 * 23, dtype=np.float32).reshape(2, 5, 23)

    np.testing.assert_array_equal(latest_right_xyz(state), state[:, -1, 9:12])


def test_latest_right_xyz_rejects_malformed_state() -> None:
    with pytest.raises(ValueError, match="unexpected proprio state shape"):
        latest_right_xyz(np.zeros((1, 5, 10), dtype=np.float32))
