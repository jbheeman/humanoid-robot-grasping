from pathlib import Path

import numpy as np
import pytest

from object_tracking.arm_tracking.gravity import UrdfGravityCompensator
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path


def gravity_model() -> UrdfGravityCompensator:
    path = default_urdf_path(Path(__file__).resolve().parents[1])
    if not path.is_file():
        pytest.skip("pinned G1 arm assets are not installed")
    return UrdfGravityCompensator(path)


def test_lightweight_gravity_model_matches_known_neutral_torque() -> None:
    model = gravity_model()

    torque = model.torque((0.0,) * 7)

    assert torque == pytest.approx(
        (-3.6433, -0.2017, -0.0002, -3.4123, 0.0388, -1.2191, 0.0),
        abs=1e-3,
    )
    assert abs(torque[3]) > 3.0  # elbow feed-forward must carry the forearm


def test_lightweight_gravity_model_matches_pinocchio_across_arm_workspace() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    lightweight = gravity_model()
    path = default_urdf_path(Path(__file__).resolve().parents[1])
    pinocchio = G1RightArmIK(path)
    configurations = (
        (0.0,) * 7,
        (0.30, -0.13, 0.0, 0.98, -0.11, 0.06, -0.04),
        (-0.50, -0.70, 0.40, 1.50, 0.20, -0.50, 0.30),
        (1.00, 0.40, -0.80, 0.20, -0.30, 0.70, -0.60),
    )

    for q in configurations:
        assert lightweight.torque(q) == pytest.approx(
            pinocchio.gravity_compensation_torque(np.asarray(q)),
            abs=1e-10,
        )


def test_lightweight_gravity_model_validates_input_and_bounds_output() -> None:
    model = gravity_model()

    with pytest.raises(ValueError, match="seven finite"):
        model.torque((0.0,) * 6)

    torque = model.torque((2.5, -2.0, 2.0, 1.8, 1.5, -1.4, 1.4))
    assert all(abs(value) <= limit for value, limit in zip(torque, model.limits_nm))
