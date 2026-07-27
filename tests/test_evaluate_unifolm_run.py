from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.evaluate_unifolm_run import (
    checkpoint_step,
    discover_checkpoints,
    validation_gate,
    weighted_ade,
)


def source_metrics(*, ade: float, visual: bool = True) -> dict:
    return {
        "model": {"ade_m": ade},
        "diagnosis_flags": {
            "visually_conditioned": visual,
            "normalized_output_saturation_over_5_percent": False,
            "action_magnitude_over_2x_target": False,
        },
    }


def test_checkpoint_discovery_requires_expected_final_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (500, 1000):
        (checkpoint_dir / f"steps_{step}_action_model.pt").write_bytes(b"weights")
    checkpoints = discover_checkpoints(tmp_path, expected_final_step=1000)
    assert [checkpoint_step(path) for path in checkpoints] == [500, 1000]
    with pytest.raises(RuntimeError, match="training incomplete"):
        discover_checkpoints(tmp_path, expected_final_step=1500)


def test_weighted_selection_metric_prioritizes_real_data() -> None:
    report = {
        "sources": {
            "g1_plush_touch_real": source_metrics(ade=0.04),
            "g1_plush_touch_sim": source_metrics(ade=0.12),
        }
    }
    assert weighted_ade(report, 0.75) == pytest.approx(0.06)


def test_validation_gate_fails_closed_on_missing_visual_conditioning() -> None:
    report = {
        "gate": {"passed": True},
        "sources": {
            "g1_plush_touch_real": source_metrics(ade=0.04, visual=False),
            "g1_plush_touch_sim": source_metrics(ade=0.05),
        },
    }
    gate = validation_gate(report)
    assert not gate["passed"]
    assert not gate["visually_conditioned"]["g1_plush_touch_real"]
