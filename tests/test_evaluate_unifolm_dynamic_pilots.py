from scripts.training.evaluate_unifolm_dynamic_pilots import score


def test_dynamic_pilot_score_prioritizes_real_validation() -> None:
    report = {
        "sources": {
            "g1_plush_touch_real": {"model": {"ade_m": 0.04}},
            "g1_plush_touch_sim": {"model": {"ade_m": 0.08}},
        }
    }
    assert score(report) == 0.05
