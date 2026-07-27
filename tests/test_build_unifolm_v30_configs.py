from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "training"
    / "build_unifolm_v30_configs.py"
)
SPEC = importlib.util.spec_from_file_location("build_v30_configs", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_v30_config_uses_active_horizon_contract(tmp_path: Path) -> None:
    statistics = tmp_path / "stats.json"
    statistics.write_text(
        json.dumps(
            {
                "representation": {
                    "normalization": {
                        "version": "per_horizon_bounds_q99_v1"
                    }
                }
            }
        )
    )
    base = {
        "datasets": {"vla_data": {}},
        "framework": {"action_model": {}},
        "trainer": {"learning_rate": {}},
    }
    config = module.build(base, statistics, tmp_path, "mixed_active")
    data = config["datasets"]["vla_data"]
    assert data["active_motion_threshold_m"] == 0.01
    assert data["inactive_keep_probability"] == 0.15
    assert data["image_aug"] is True
    assert len(config["framework"]["action_model"]["loss_horizon_weights"]) == 25
    assert config["trainer"]["learning_rate"]["action_model"] == 3e-5
