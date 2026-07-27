import numpy as np

from scripts.training.diagnose_unifolm_predictions import (
    active_motion_mask,
    execution_horizon_summary,
    summarize_errors,
)


def test_error_summary_excludes_padded_horizon_targets() -> None:
    errors = np.asarray(((1.0, 2.0, 100.0), (3.0, 100.0, 100.0)))
    valid = np.asarray(((True, True, False), (True, False, False)))

    summary = summarize_errors(errors, valid)

    assert summary["ade_m"] == 2.0
    assert summary["fde_m"] == 2.5
    assert summary["per_horizon_mean_m"][:2] == [2.0, 2.0]
    assert summary["per_horizon_mean_m"][2] is None
    assert summary["per_horizon_valid_count"] == [2, 1, 0]


def test_active_motion_mask_excludes_hold_dominated_targets() -> None:
    current = np.zeros((1, 3, 3))
    target = current.copy()
    target[0, 1, 0] = 0.02
    target[0, 2, 0] = 0.03
    valid = np.asarray([[True, True, False]])
    assert active_motion_mask(target, current, valid).tolist() == [
        [False, True, False]
    ]


def test_execution_horizon_scores_first_nonexpired_waypoint() -> None:
    errors = np.arange(25, dtype=float)[None, :]
    valid = np.ones_like(errors, dtype=bool)
    summary = execution_horizon_summary(errors, valid, latency_s=0.37)
    assert summary["waypoint_index"] == 11
    assert summary["mean_error_m"] == 11.0
