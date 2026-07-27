from __future__ import annotations

import json
from pathlib import Path

from scripts.training.build_unifolm_motion_history_config import build_config


def test_motion_config_uses_history_relative_actions_and_right_hand(tmp_path: Path) -> None:
    statistics = tmp_path / "relative.json"
    statistics.write_text(
        json.dumps(
            {
                "representation": {
                    "version": "anchored_relative_pose23_v1",
                    "target_alignment": {
                        "version": "achieved_future_state_v1",
                        "lookahead_frames": 1,
                    },
                }
            }
        )
    )
    base = {
        "framework": {"action_model": {}},
        "datasets": {"vla_data": {}},
        "trainer": {"learning_rate": {}},
    }
    config = build_config(
        base,
        data_root=tmp_path / "rlds",
        statistics=statistics,
        max_train_steps=4000,
    )
    data = config["datasets"]["vla_data"]
    assert (data["window_size"], data["observation_stride"]) == (5, 3)
    assert data["image_aug"] is True
    assert config["framework"]["action_model"]["use_relative_action"] is True
    weights = config["framework"]["action_model"]["loss_dimension_weights"]
    assert weights[9:18] == [1.0] * 9
    assert sum(weights) == 9.0
