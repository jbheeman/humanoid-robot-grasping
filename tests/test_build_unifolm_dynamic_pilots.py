from scripts.training.build_unifolm_dynamic_pilots import (
    build_variants,
    right_pose_loss_weights,
)


def base_config() -> dict:
    return {
        "framework": {
            "action_model": {},
        },
        "datasets": {
            "vla_data": {
                "window_size": 1,
                "observation_stride": 1,
            }
        },
        "trainer": {
            "max_train_steps": 10_000,
            "num_warmup_steps": 200,
            "save_interval": 500,
            "eval_interval": 500,
            "learning_rate": {
                "base": 1e-4,
                "action_model": 1e-4,
            },
            "is_resume": True,
            "resume_epoch": 1,
            "resume_step": 100,
        },
        "run_id": "base",
    }


def test_right_pose_loss_weights_focus_only_the_controlled_effector() -> None:
    weights = right_pose_loss_weights()
    assert len(weights) == 23
    assert weights[9:18] == [1.0] * 9
    assert weights[:9] == [0.05] * 9
    assert weights[18:] == [0.05] * 5


def test_temporal_pilot_spans_four_tenths_of_a_second() -> None:
    variants = build_variants(base_config(), max_train_steps=750)
    temporal = variants["t5s3-right9"]
    assert temporal["datasets"]["vla_data"]["window_size"] == 5
    assert temporal["datasets"]["vla_data"]["observation_stride"] == 3
    assert temporal["trainer"]["max_train_steps"] == 750
    assert temporal["trainer"]["learning_rate"]["action_model"] == 3e-5
    assert temporal["trainer"]["is_resume"] is False


def test_control_variant_keeps_original_unweighted_loss() -> None:
    variants = build_variants(base_config(), max_train_steps=500)
    assert "loss_dimension_weights" not in variants["t1-all23"]["framework"][
        "action_model"
    ]
