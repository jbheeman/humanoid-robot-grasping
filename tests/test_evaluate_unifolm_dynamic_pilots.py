import pytest

from scripts.training.evaluate_unifolm_dynamic_pilots import (
    assign_variants_to_gpus,
    score,
)


def test_dynamic_pilot_score_prioritizes_real_validation() -> None:
    report = {
        "sources": {
            "g1_plush_touch_real": {"model": {"ade_m": 0.04}},
            "g1_plush_touch_sim": {"model": {"ade_m": 0.08}},
        }
    }
    assert score(report) == 0.05


def test_dynamic_pilots_are_balanced_across_two_gpus() -> None:
    queues = assign_variants_to_gpus(
        ("t1-all23", "t1-right9", "t5s3-right9"),
        (0, 1),
    )

    assert queues == {
        0: ("t5s3-right9",),
        1: ("t1-all23", "t1-right9"),
    }


@pytest.mark.parametrize("gpus", ((), (0, 0), (-1, 0)))
def test_dynamic_pilot_gpu_assignments_reject_invalid_indices(
    gpus: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        assign_variants_to_gpus(("t1-all23",), gpus)
