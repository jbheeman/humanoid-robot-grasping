import numpy as np

from scripts.training.diagnose_unifolm_predictions import summarize_errors


def test_error_summary_excludes_padded_horizon_targets() -> None:
    errors = np.asarray(((1.0, 2.0, 100.0), (3.0, 100.0, 100.0)))
    valid = np.asarray(((True, True, False), (True, False, False)))

    summary = summarize_errors(errors, valid)

    assert summary["ade_m"] == 2.0
    assert summary["fde_m"] == 2.5
    assert summary["per_horizon_mean_m"][:2] == [2.0, 2.0]
    assert summary["per_horizon_mean_m"][2] is None
    assert summary["per_horizon_valid_count"] == [2, 1, 0]
