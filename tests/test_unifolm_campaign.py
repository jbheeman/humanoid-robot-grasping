from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "training" / "unifolm_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("unifolm_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def report(checkpoint: Path, real_ade: float, sim_ade: float) -> dict:
    def source(ade: float, baseline: float) -> dict:
        return {
            "model": {
                "ade_m": ade,
                "fde_m": ade * 1.1,
                "median_m": ade * 0.9,
                "p95_m": ade * 1.5,
            },
            "best_baseline_ade_m": baseline,
            "normalized_output_saturation_fraction": 0.01,
            "predicted_to_target_displacement_ratio": 1.0,
            "diagnosis_flags": {"visually_conditioned": True},
        }

    return {
        "checkpoint": str(checkpoint),
        "action_representation": "relative_pose23",
        "window_size": 2,
        "observation_stride": 1,
        "sources": {
            "g1_plush_touch_real": source(real_ade, 0.02),
            "g1_plush_touch_sim": source(sim_ade, 0.04),
        },
    }


def test_report_metrics_are_baseline_normalized() -> None:
    objective, raw, sources = campaign.report_metrics(
        report(Path("/tmp/checkpoint.pt"), 0.02, 0.04)
    )
    assert objective == 1.0
    assert raw == 0.025
    assert sources["g1_plush_touch_real"]["normalized_ade"] == 1.0


def test_backfill_selects_strong_report_per_run(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "run-a"
    diagnostics.mkdir(parents=True)
    checkpoint = tmp_path / "runs" / "unifolm_plush_touch" / "run-a" / "x.pt"
    fast = diagnostics / "step_1000_fast_val.json"
    strong = diagnostics / "selected_step_1000_strong_val.json"
    fast.write_text(json.dumps(report(checkpoint, 0.08, 0.08)))
    strong.write_text(json.dumps(report(checkpoint, 0.03, 0.04)))
    db = campaign.CampaignDB(tmp_path / "campaign.sqlite3")
    assert campaign.backfill(db, tmp_path / "diagnostics", "test") == 1
    row = db.comparable_attempts()[0]
    assert row["weighted_ade_m"] == 0.0325
    assert "strong_val" in row["detail"]


def test_next_parameters_uses_existing_checkpoint_and_varies(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    base.write_bytes(b"x")
    db = campaign.CampaignDB(tmp_path / "campaign.sqlite3")
    first = campaign.next_parameters(db, "test", base, 4000)
    assert first["parent_checkpoint"] == str(base)
    db.upsert_attempt(first)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report(base, 0.03, 0.04)))
    db.record_report(first["attempt_id"], report_path)
    second = campaign.next_parameters(db, "test", base, 4000)
    assert second["ordinal"] == 2
    assert second["learning_rate"] != first["learning_rate"]
    assert second["parent_checkpoint"] == str(base)


def test_resume_control_clears_both_markers(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {"workspace": tmp_path, "campaign": "test", "command": "resume"},
    )()
    directory = tmp_path / "runs" / "automation" / "test"
    directory.mkdir(parents=True)
    for name in ("STOP_NOW", "STOP_AFTER_ATTEMPT"):
        (directory / name).write_text("stop")
    assert campaign.control(args) == 0
    assert not (directory / "STOP_NOW").exists()
    assert not (directory / "STOP_AFTER_ATTEMPT").exists()


def test_training_state_requires_complete_files(tmp_path: Path) -> None:
    state = (
        tmp_path
        / "checkpoints"
        / "steps_1000_training_state"
        / "rank0"
    )
    state.mkdir(parents=True)
    (state / "x_model_states.pt").write_bytes(b"x" * (1024 * 1024 + 1))
    assert campaign.verify_training_states(tmp_path) == []
    (state / "x_optim_states.pt").write_bytes(b"x" * (1024 * 1024 + 1))
    verified = campaign.verify_training_states(tmp_path)
    assert verified == [state.parent]
    assert (state.parent / "COMPLETE.json").is_file()
