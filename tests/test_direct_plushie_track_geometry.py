"""Regression coverage for the factory G1 D435I bearing model."""

from __future__ import annotations

import importlib.util
import sys
from math import isclose
from pathlib import Path


def _tracker_module():
    path = Path(__file__).resolve().parents[1] / "scripts/robot/direct_plushie_track.py"
    spec = importlib.util.spec_from_file_location("direct_plushie_track_geometry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_d435i_principal_ray_uses_the_urdf_head_pitch() -> None:
    tracker = _tracker_module()
    intrinsics = tracker._scale_rgb_intrinsics(960, 540)

    azimuth, elevation = tracker._torso_bearing_from_rgb_pixel((490.993, 292.158), intrinsics)

    assert isclose(azimuth, 0.0, abs_tol=1e-8)
    assert isclose(elevation, -0.8307767, abs_tol=1e-7)


def test_d435i_left_and_right_pixels_have_opposing_torso_bearings() -> None:
    tracker = _tracker_module()
    intrinsics = tracker._scale_rgb_intrinsics(960, 540)

    left_azimuth, _ = tracker._torso_bearing_from_rgb_pixel((0.0, 292.158), intrinsics)
    right_azimuth, _ = tracker._torso_bearing_from_rgb_pixel((960.0, 292.158), intrinsics)

    assert left_azimuth > 0.0
    assert right_azimuth < 0.0
