from __future__ import annotations

import numpy as np

from object_tracking.arm_tracking.geometry import CameraIntrinsics
from object_tracking.arm_tracking.localization import LocalizationObservation, solve_localization


INTRINSICS = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)


def _sample(name: str, torso: tuple[float, float, float]) -> LocalizationObservation:
    # Synthetic camera is coincident with torso, so the explicit pinhole
    # equation produces the known torso coordinates exactly.
    x, y, z = torso
    return LocalizationObservation(name, (500.0 * x / z + 480.0, 500.0 * y / z + 270.0), z, torso)


def test_localization_solves_and_validates_held_out_positions() -> None:
    samples = [
        _sample("fit-center", (0.0, 0.0, 0.8)),
        _sample("fit-left", (0.1, 0.2, 0.9)),
        _sample("fit-right", (0.2, -0.2, 1.0)),
        _sample("fit-high", (0.35, 0.1, 1.1)),
        _sample("fit-low", (-0.2, -0.1, 0.7)),
        _sample("fit-far", (0.0, 0.2, 1.3)),
        _sample("val-left", (0.1, 0.3, 0.85)),
        _sample("val-right", (-0.1, -0.3, 0.95)),
        _sample("val-high", (0.45, 0.1, 1.0)),
        _sample("val-low", (-0.35, -0.1, 0.75)),
    ]
    report = solve_localization(
        samples,
        INTRINSICS,
        fit_names=[sample.name for sample in samples[:6]],
        validation_names=[sample.name for sample in samples[6:]],
    )
    assert report.passed
    assert np.isclose(report.median_error_m, 0.0)
    assert report.left_right_ok
    assert report.up_down_ok


def test_localization_rejects_lateral_flip() -> None:
    samples = [
        _sample("fit-1", (0.0, 0.0, 0.8)),
        _sample("fit-2", (0.2, 0.2, 1.0)),
        _sample("fit-3", (0.3, -0.2, 1.2)),
        _sample("val-left", (0.1, 0.2, 0.9)),
        _sample("val-right", (0.1, -0.2, 0.9)),
        _sample("val-high", (0.3, 0.1, 1.0)),
        _sample("val-low", (-0.2, -0.1, 1.0)),
    ]
    # Swapping the truth labels causes the held-out side check to fail even if
    # an error threshold is later relaxed for diagnosis.
    samples[3] = LocalizationObservation("val-left", samples[3].pixel_xy, samples[3].depth_m, (0.1, -0.2, 0.9))
    report = solve_localization(
        samples, INTRINSICS,
        fit_names=[sample.name for sample in samples[:3]],
        validation_names=[sample.name for sample in samples[3:]],
    )
    assert not report.left_right_ok
    assert not report.passed
