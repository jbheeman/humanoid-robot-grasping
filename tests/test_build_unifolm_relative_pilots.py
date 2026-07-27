from pathlib import Path

from scripts.training.build_unifolm_relative_pilots import (
    build_variants,
    right_only_loss_weights,
)


def base_config() -> dict:
    return {
        "framework": {"action_model": {}},
        "datasets": {
            "vla_data": {
                "window_size": 1,
                "observation_stride": 1,
            }
        },
        "trainer": {
            "learning_rate": {"base": 1e-4, "action_model": 1e-4},
            "is_resume": True,
        },
        "run_id": "base",
    }


def test_right_only_weights_mask_held_dimensions() -> None:
    weights = right_only_loss_weights()

    assert weights[9:18] == [1.0] * 9
    assert sum(weights) == 9.0


def test_relative_and_absolute_pilots_are_temporally_matched() -> None:
    variants = build_variants(
        base_config(),
        relative_statistics=Path("/tmp/relative.json"),
        max_train_steps=750,
    )

    assert set(variants) == {
        "absolute-t1-right9",
        "absolute-t5s3-right9",
        "relative-t1-right9",
        "relative-t5s3-right9",
    }
    for temporal in ("t1", "t5s3"):
        absolute = variants[f"absolute-{temporal}-right9"]
        relative = variants[f"relative-{temporal}-right9"]
        assert absolute["trainer"]["max_train_steps"] == 750
        assert absolute["datasets"]["vla_data"]["window_size"] == relative[
            "datasets"
        ]["vla_data"]["window_size"]
        assert absolute["framework"]["action_model"]["use_relative_action"] is False
        assert relative["framework"]["action_model"]["use_relative_action"] is True
