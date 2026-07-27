from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "training"
    / "build_v30_horizon_statistics.py"
)
SPEC = importlib.util.spec_from_file_location("build_v30_stats", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_promotes_per_horizon_arrays() -> None:
    action = {
        field: [0.0, 1.0]
        for field in ("min", "max", "mean", "std", "q01", "q99")
    }
    source = {
        "action": action,
        "proprio": action,
        "provenance": {
            "per_horizon": {
                "0": {"action": action},
                "1": {
                    "action": {
                        field: [2.0, 3.0]
                        for field in ("min", "max", "mean", "std", "q01", "q99")
                    }
                },
            }
        },
        "representation": {"horizon": 2},
    }
    result = module.build(source, 2)
    assert result["horizon_action"]["q99"] == [[0.0, 1.0], [2.0, 3.0]]
    assert (
        result["representation"]["normalization"]["version"]
        == "per_horizon_bounds_q99_v1"
    )
