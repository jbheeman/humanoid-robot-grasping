from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "training"
    / "audit_v30_motion_distribution.py"
)
SPEC = importlib.util.spec_from_file_location("audit_v30", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_episode_motion_uses_future_relative_displacement() -> None:
    xyz = np.asarray([[0, 0, 0], [0.01, 0, 0], [0.03, 0, 0]], dtype=float)
    displacement, valid = module.episode_motion(xyz, 2)
    assert np.allclose(displacement[0], [0.01, 0.03])
    assert valid.tolist() == [[True, True], [True, False], [False, False]]
