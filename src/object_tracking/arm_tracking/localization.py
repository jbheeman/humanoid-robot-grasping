"""Offline, no-actuation camera-to-torso localization validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .calibration import solve_rigid_transform
from .geometry import CameraIntrinsics, RigidTransform


@dataclass(frozen=True)
class LocalizationObservation:
    """One paired RGB/depth observation with independently measured truth."""

    name: str
    pixel_xy: tuple[float, float]
    depth_m: float
    torso_truth_m: tuple[float, float, float]

    def optical_point(self, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Use the documented pinhole equation at the RGB box centre."""

        u, v = self.pixel_xy
        z = self.depth_m
        if not self.name or not np.all(np.isfinite((u, v, z, *self.torso_truth_m))) or z <= 0:
            raise ValueError("localization observation contains invalid values")
        return np.array(
            ((u - intrinsics.ppx) * z / intrinsics.fx, (v - intrinsics.ppy) * z / intrinsics.fy, z),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LocalizationReport:
    transform: RigidTransform
    fit_names: tuple[str, ...]
    validation_names: tuple[str, ...]
    validation_errors_m: tuple[float, ...]
    median_error_m: float
    p95_error_m: float
    left_right_ok: bool
    up_down_ok: bool

    @property
    def passed(self) -> bool:
        return (
            self.median_error_m <= 0.05
            and self.p95_error_m <= 0.08
            and self.left_right_ok
            and self.up_down_ok
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "optical_to_torso": self.transform.to_dict(),
            "fit_names": list(self.fit_names),
            "validation_names": list(self.validation_names),
            "validation_errors_m": list(self.validation_errors_m),
            "median_error_m": self.median_error_m,
            "p95_error_m": self.p95_error_m,
            "left_right_ok": self.left_right_ok,
            "up_down_ok": self.up_down_ok,
            "passed": self.passed,
            "limits_m": {"median": 0.05, "p95": 0.08},
        }


def _direction_preserved(predicted: np.ndarray, truth: np.ndarray, axis: int) -> bool:
    """Reject any held-out point whose lateral/vertical side is inverted.

    Directions are relative to the median measured target position, so the
    check is invariant to the physical mounting translation.
    """

    origin = float(np.median(truth[:, axis]))
    known = truth[:, axis] - origin
    estimate = predicted[:, axis] - origin
    # A point deliberately placed at the centre does not establish a side.
    meaningful = np.abs(known) >= 0.02
    return bool(np.all(known[meaningful] * estimate[meaningful] > 0)) if np.any(meaningful) else False


def solve_localization(
    observations: Sequence[LocalizationObservation],
    intrinsics: CameraIntrinsics,
    *,
    fit_names: Iterable[str],
    validation_names: Iterable[str],
) -> LocalizationReport:
    """Fit from named samples and evaluate only the held-out samples."""

    by_name = {sample.name: sample for sample in observations}
    if len(by_name) != len(observations):
        raise ValueError("localization observation names must be unique")
    fit_names = tuple(fit_names)
    validation_names = tuple(validation_names)
    if len(fit_names) < 3 or len(validation_names) < 4:
        raise ValueError("at least three fit samples and four held-out validation samples are required")
    if set(fit_names) & set(validation_names):
        raise ValueError("fit and validation samples must be disjoint")
    try:
        fit = [by_name[name] for name in fit_names]
        validation = [by_name[name] for name in validation_names]
    except KeyError as exc:
        raise ValueError(f"unknown localization sample {exc.args[0]!r}") from exc

    transform = solve_rigid_transform(
        (sample.optical_point(intrinsics) for sample in fit),
        (sample.torso_truth_m for sample in fit),
    ).transform
    predicted = transform.apply(np.asarray([sample.optical_point(intrinsics) for sample in validation]))
    truth = np.asarray([sample.torso_truth_m for sample in validation], dtype=np.float64)
    errors = np.linalg.norm(predicted - truth, axis=1)
    return LocalizationReport(
        transform=transform,
        fit_names=fit_names,
        validation_names=validation_names,
        validation_errors_m=tuple(float(value) for value in errors),
        median_error_m=float(np.median(errors)),
        p95_error_m=float(np.percentile(errors, 95)),
        left_right_ok=_direction_preserved(predicted, truth, 1),
        up_down_ok=_direction_preserved(predicted, truth, 2),
    )
