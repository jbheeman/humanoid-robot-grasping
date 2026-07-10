"""Calibration schema, rigid solve, validation, tuning, and atomic YAML I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from .geometry import CameraIntrinsics, RigidTransform, WorkspaceBounds


SCHEMA_VERSION = 1
MIN_SOLVE_POSES = 8
MIN_VALIDATION_POSES = 4
MAX_VALIDATION_RMSE_M = 0.025
MAX_VALIDATION_ERROR_M = 0.050
MAX_ORIENTATION_ERROR_DEG = 3.0
MAX_WAIST_DEVIATION_DEG = 3.0
MAX_REGISTRATION_MEDIAN_ERROR_PX = 3.0
MAX_REGISTRATION_ERROR_PX = 6.0


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    fps: int
    format: str

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.fps) <= 0 or not self.format:
            raise CalibrationError("stream profile fields must be positive and non-empty")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "fps", int(self.fps))

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StreamProfile":
        return cls(
            int(value["width"]), int(value["height"]), int(value["fps"]), str(value["format"])
        )


@dataclass(frozen=True)
class CalibrationPose:
    name: str
    optical_point_m: tuple[float, float, float]
    torso_point_m: tuple[float, float, float]
    position_error_m: float | None = None
    orientation_error_deg: float | None = None
    source: str = "manual"

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[float, float, float]] = {}
        for name, value in (
            ("optical_point_m", self.optical_point_m),
            ("torso_point_m", self.torso_point_m),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise CalibrationError(f"{name} must contain three finite values")
            normalized[name] = tuple(float(item) for item in array)
        object.__setattr__(self, "optical_point_m", normalized["optical_point_m"])
        object.__setattr__(self, "torso_point_m", normalized["torso_point_m"])
        if not self.name:
            raise CalibrationError("calibration pose name is required")
        if self.source not in {"apriltag", "manual"}:
            raise CalibrationError("calibration pose source must be 'apriltag' or 'manual'")
        if self.position_error_m is not None and self.position_error_m < 0:
            raise CalibrationError("position error cannot be negative")
        if self.orientation_error_deg is not None and self.orientation_error_deg < 0:
            raise CalibrationError("orientation error cannot be negative")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "optical_point_m": list(self.optical_point_m),
            "torso_point_m": list(self.torso_point_m),
            "source": self.source,
        }
        if self.position_error_m is not None:
            result["position_error_m"] = self.position_error_m
        if self.orientation_error_deg is not None:
            result["orientation_error_deg"] = self.orientation_error_deg
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationPose":
        return cls(
            name=str(value["name"]),
            optical_point_m=tuple(float(item) for item in value["optical_point_m"]),
            torso_point_m=tuple(float(item) for item in value["torso_point_m"]),
            position_error_m=(
                None if value.get("position_error_m") is None else float(value["position_error_m"])
            ),
            orientation_error_deg=(
                None
                if value.get("orientation_error_deg") is None
                else float(value["orientation_error_deg"])
            ),
            source=str(value.get("source", "manual")),
        )


@dataclass(frozen=True)
class ResidualSummary:
    rmse_m: float
    max_error_m: float
    max_orientation_error_deg: float

    def __post_init__(self) -> None:
        values = (self.rmse_m, self.max_error_m, self.max_orientation_error_deg)
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise CalibrationError("residual summary values must be finite and non-negative")
        object.__setattr__(self, "rmse_m", float(self.rmse_m))
        object.__setattr__(self, "max_error_m", float(self.max_error_m))
        object.__setattr__(self, "max_orientation_error_deg", float(self.max_orientation_error_deg))

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse_m": self.rmse_m,
            "max_error_m": self.max_error_m,
            "max_orientation_error_deg": self.max_orientation_error_deg,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResidualSummary":
        return cls(
            float(value["rmse_m"]),
            float(value["max_error_m"]),
            float(value.get("max_orientation_error_deg", 0.0)),
        )


@dataclass(frozen=True)
class RegistrationResiduals:
    median_error_px: float
    max_error_px: float
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not np.all(np.isfinite((self.median_error_px, self.max_error_px)))
            or min(self.median_error_px, self.max_error_px) < 0
            or self.sample_count < 1
        ):
            raise CalibrationError("registration residuals must be finite and non-negative")
        object.__setattr__(self, "median_error_px", float(self.median_error_px))
        object.__setattr__(self, "max_error_px", float(self.max_error_px))
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "median_error_px": self.median_error_px,
            "max_error_px": self.max_error_px,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistrationResiduals":
        return cls(
            float(value["median_error_px"]),
            float(value["max_error_px"]),
            int(value["sample_count"]),
        )


@dataclass(frozen=True)
class Calibration:
    calibration_id: str
    calibration_hash: str
    created_at: str
    camera_serial: str
    camera_firmware: str
    rgb_profile: StreamProfile
    depth_profile: StreamProfile
    rgb_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    depth_scale: float
    depth_to_rgb: RigidTransform
    optical_to_torso: RigidTransform
    waist_reference_rad: tuple[float, float, float]
    tag_to_wrist: RigidTransform
    workspace: WorkspaceBounds
    solve_poses: tuple[CalibrationPose, ...]
    validation_poses: tuple[CalibrationPose, ...]
    solve_residuals: ResidualSummary
    validation_residuals: ResidualSummary
    registration_residuals: RegistrationResiduals | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        waist = tuple(float(item) for item in self.waist_reference_rad)
        object.__setattr__(self, "waist_reference_rad", waist)
        object.__setattr__(self, "depth_scale", float(self.depth_scale))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "created_at": self.created_at,
            "camera": {
                "serial": self.camera_serial,
                "firmware": self.camera_firmware,
                "rgb_profile": self.rgb_profile.to_dict(),
                "depth_profile": self.depth_profile.to_dict(),
                "rgb_intrinsics": self.rgb_intrinsics.to_dict(),
                "depth_intrinsics": self.depth_intrinsics.to_dict(),
                "depth_scale": self.depth_scale,
                "depth_to_rgb": self.depth_to_rgb.to_dict(),
            },
            "optical_to_torso": self.optical_to_torso.to_dict(),
            "waist_reference_rad": list(self.waist_reference_rad),
            "tag_to_wrist": self.tag_to_wrist.to_dict(),
            "workspace": {
                "minimum": self.workspace.minimum.tolist(),
                "maximum": self.workspace.maximum.tolist(),
            },
            "solve_poses": [pose.to_dict() for pose in self.solve_poses],
            "validation_poses": [pose.to_dict() for pose in self.validation_poses],
            "solve_residuals": self.solve_residuals.to_dict(),
            "validation_residuals": self.validation_residuals.to_dict(),
        }
        if self.registration_residuals is not None:
            result["registration_residuals"] = self.registration_residuals.to_dict()
        if include_hash:
            result["calibration_hash"] = self.calibration_hash
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        check_hash: bool = True,
        require_complete: bool = True,
    ) -> "Calibration":
        try:
            camera = value["camera"]
            workspace = value["workspace"]
            calibration = cls(
                schema_version=int(value["schema_version"]),
                calibration_id=str(value["calibration_id"]),
                calibration_hash=str(value["calibration_hash"]),
                created_at=str(value["created_at"]),
                camera_serial=str(camera["serial"]),
                camera_firmware=str(camera["firmware"]),
                rgb_profile=StreamProfile.from_dict(camera["rgb_profile"]),
                depth_profile=StreamProfile.from_dict(camera["depth_profile"]),
                rgb_intrinsics=CameraIntrinsics.from_dict(camera["rgb_intrinsics"]),
                depth_intrinsics=CameraIntrinsics.from_dict(camera["depth_intrinsics"]),
                depth_scale=float(camera["depth_scale"]),
                depth_to_rgb=RigidTransform.from_dict(camera["depth_to_rgb"]),
                optical_to_torso=RigidTransform.from_dict(value["optical_to_torso"]),
                waist_reference_rad=tuple(float(item) for item in value["waist_reference_rad"]),
                tag_to_wrist=RigidTransform.from_dict(value["tag_to_wrist"]),
                workspace=WorkspaceBounds(workspace["minimum"], workspace["maximum"]),
                solve_poses=tuple(CalibrationPose.from_dict(item) for item in value["solve_poses"]),
                validation_poses=tuple(
                    CalibrationPose.from_dict(item) for item in value["validation_poses"]
                ),
                solve_residuals=ResidualSummary.from_dict(value["solve_residuals"]),
                validation_residuals=ResidualSummary.from_dict(value["validation_residuals"]),
                registration_residuals=(
                    None
                    if value.get("registration_residuals") is None
                    else RegistrationResiduals.from_dict(value["registration_residuals"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration schema: {exc}") from exc
        calibration.validate_structure(require_complete=require_complete, check_hash=check_hash)
        return calibration

    def computed_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def with_computed_hash(self) -> "Calibration":
        return replace(self, calibration_hash=self.computed_hash())

    def validate_structure(self, *, require_complete: bool = True, check_hash: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CalibrationError(f"unsupported calibration schema {self.schema_version}")
        try:
            uuid.UUID(self.calibration_id)
        except ValueError as exc:
            raise CalibrationError("calibration_id must be a UUID") from exc
        if not self.camera_serial or not self.camera_firmware:
            raise CalibrationError("camera serial and firmware are required")
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CalibrationError("created_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise CalibrationError("created_at must include a timezone")
        if not 0 < self.depth_scale < 1:
            raise CalibrationError("depth scale must be between zero and one")
        if len(self.waist_reference_rad) != 3 or not np.all(np.isfinite(self.waist_reference_rad)):
            raise CalibrationError("waist reference must contain three finite radians")
        if (
            self.rgb_intrinsics.width != self.rgb_profile.width
            or self.rgb_intrinsics.height != self.rgb_profile.height
        ):
            raise CalibrationError("RGB intrinsics and profile dimensions differ")
        if (
            self.depth_intrinsics.width != self.depth_profile.width
            or self.depth_intrinsics.height != self.depth_profile.height
        ):
            raise CalibrationError("depth intrinsics and profile dimensions differ")
        if require_complete and len(self.solve_poses) < MIN_SOLVE_POSES:
            raise CalibrationError(f"at least {MIN_SOLVE_POSES} solve poses are required")
        if require_complete and len(self.validation_poses) < MIN_VALIDATION_POSES:
            raise CalibrationError(
                f"at least {MIN_VALIDATION_POSES} held-out validation poses are required"
            )
        if require_complete:
            solve_points = np.asarray(
                [pose.optical_point_m for pose in self.solve_poses], dtype=np.float64
            )
            validation_points = np.asarray(
                [pose.optical_point_m for pose in self.validation_poses], dtype=np.float64
            )
            if len(np.unique(solve_points, axis=0)) != len(solve_points):
                raise CalibrationError("solve poses must contain unique optical points")
            if np.linalg.matrix_rank(solve_points - solve_points.mean(axis=0)) < 2:
                raise CalibrationError("solve poses are not spatially distributed")
            if np.linalg.matrix_rank(validation_points - validation_points.mean(axis=0)) < 2:
                raise CalibrationError("validation poses are not spatially distributed")
            solve_names = {pose.name for pose in self.solve_poses}
            validation_names = {pose.name for pose in self.validation_poses}
            if solve_names & validation_names:
                raise CalibrationError("validation poses must be held out from solve poses")
        if check_hash and self.calibration_hash != self.computed_hash():
            raise CalibrationError("calibration hash does not match calibration contents")

    def validate_for_execution(
        self,
        *,
        camera_serial: str,
        rgb_profile: StreamProfile,
        depth_profile: StreamProfile,
        waist_rad: Sequence[float],
        calibration_id: str | None = None,
        camera_firmware: str | None = None,
        max_waist_deviation_deg: float = MAX_WAIST_DEVIATION_DEG,
    ) -> None:
        self.validate_structure()
        if camera_serial != self.camera_serial:
            raise CalibrationError("camera serial does not match calibration")
        if camera_firmware is not None and camera_firmware != self.camera_firmware:
            raise CalibrationError("camera firmware does not match calibration")
        if rgb_profile != self.rgb_profile or depth_profile != self.depth_profile:
            raise CalibrationError("active stream profiles do not match calibration")
        if calibration_id is not None and calibration_id != self.calibration_id:
            raise CalibrationError("active calibration ID does not match")
        waist = np.asarray(waist_rad, dtype=np.float64)
        reference = np.asarray(self.waist_reference_rad, dtype=np.float64)
        if waist.shape != (3,) or not np.all(np.isfinite(waist)):
            raise CalibrationError("live waist state must contain three finite values")
        maximum_deviation = float(np.max(np.abs(np.rad2deg(waist - reference))))
        if maximum_deviation > max_waist_deviation_deg:
            raise CalibrationError(
                f"waist deviates {maximum_deviation:.2f} degrees from calibration"
            )
        if self.validation_residuals.rmse_m > MAX_VALIDATION_RMSE_M:
            raise CalibrationError("validation RMSE exceeds 25 mm")
        if self.validation_residuals.max_error_m > MAX_VALIDATION_ERROR_M:
            raise CalibrationError("validation maximum error exceeds 50 mm")
        if self.validation_residuals.max_orientation_error_deg > MAX_ORIENTATION_ERROR_DEG:
            raise CalibrationError("validation orientation error exceeds 3 degrees")
        if self.registration_residuals is None:
            raise CalibrationError("RGB/depth registration has not been validated")
        if self.registration_residuals.median_error_px > MAX_REGISTRATION_MEDIAN_ERROR_PX:
            raise CalibrationError("registration median reprojection error exceeds 3 px")
        if self.registration_residuals.max_error_px > MAX_REGISTRATION_ERROR_PX:
            raise CalibrationError("registration maximum reprojection error exceeds 6 px")


@dataclass(frozen=True)
class RigidSolveResult:
    transform: RigidTransform
    errors_m: np.ndarray
    residuals: ResidualSummary


def validate_registration_correspondences(
    expected_rgb_pixels: Iterable[Iterable[float]],
    measured_depth_pixels: Iterable[Iterable[float]],
) -> RegistrationResiduals:
    expected = np.asarray(tuple(tuple(pixel) for pixel in expected_rgb_pixels), dtype=np.float64)
    measured = np.asarray(tuple(tuple(pixel) for pixel in measured_depth_pixels), dtype=np.float64)
    if expected.shape != measured.shape or expected.ndim != 2 or expected.shape[1] != 2:
        raise CalibrationError("registration correspondences must both have shape (N, 2)")
    if len(expected) < 4 or not np.all(np.isfinite(expected)) or not np.all(np.isfinite(measured)):
        raise CalibrationError("at least four finite registration correspondences are required")
    errors = np.linalg.norm(expected - measured, axis=1)
    return RegistrationResiduals(
        median_error_px=float(np.median(errors)),
        max_error_px=float(np.max(errors)),
        sample_count=len(errors),
    )


def solve_rigid_transform(
    optical_points_m: Iterable[Iterable[float]],
    torso_points_m: Iterable[Iterable[float]],
    *,
    orientation_errors_deg: Iterable[float] = (),
) -> RigidSolveResult:
    """Solve optical-to-torso rigid alignment with the Kabsch algorithm."""

    source = np.asarray(tuple(tuple(point) for point in optical_points_m), dtype=np.float64)
    target = np.asarray(tuple(tuple(point) for point in torso_points_m), dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise CalibrationError("source and target correspondences must both have shape (N, 3)")
    if len(source) < 3 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise CalibrationError("at least three finite correspondences are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, singular_values, vt = np.linalg.svd(covariance)
    if singular_values[1] <= 1e-10:
        raise CalibrationError("calibration correspondences are collinear")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = RigidTransform(rotation, translation)
    errors = np.linalg.norm(transform.apply(source) - target, axis=1)
    orientation_errors = tuple(float(item) for item in orientation_errors_deg)
    residuals = ResidualSummary(
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
        max_error_m=float(np.max(errors)),
        max_orientation_error_deg=max(orientation_errors, default=0.0),
    )
    return RigidSolveResult(transform, errors, residuals)


def validate_correspondences(
    transform: RigidTransform,
    poses: Sequence[CalibrationPose],
) -> ResidualSummary:
    if not poses:
        raise CalibrationError("validation poses are required")
    source = np.asarray([pose.optical_point_m for pose in poses], dtype=np.float64)
    target = np.asarray([pose.torso_point_m for pose in poses], dtype=np.float64)
    errors = np.linalg.norm(transform.apply(source) - target, axis=1)
    orientation = [
        pose.orientation_error_deg for pose in poses if pose.orientation_error_deg is not None
    ]
    return ResidualSummary(
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
        max_error_m=float(np.max(errors)),
        max_orientation_error_deg=max(orientation, default=0.0),
    )


def tune_transform(
    transform: RigidTransform,
    *,
    xyz_delta_m: Sequence[float] = (0.0, 0.0, 0.0),
    rpy_delta_deg: Sequence[float] = (0.0, 0.0, 0.0),
    max_translation_m: float = 0.05,
    max_rotation_deg: float = 5.0,
) -> RigidTransform:
    translation_delta = np.asarray(xyz_delta_m, dtype=np.float64)
    rotation_delta = np.asarray(rpy_delta_deg, dtype=np.float64)
    if translation_delta.shape != (3,) or rotation_delta.shape != (3,):
        raise CalibrationError("tuning deltas must each contain three values")
    if np.any(np.abs(translation_delta) > max_translation_m):
        raise CalibrationError("translation tuning exceeds configured bound")
    if np.any(np.abs(rotation_delta) > max_rotation_deg):
        raise CalibrationError("rotation tuning exceeds configured bound")
    adjustment = RigidTransform.from_xyz_rpy(translation_delta, np.deg2rad(rotation_delta))
    return transform.then(adjustment)


def new_calibration_id() -> str:
    return str(uuid.uuid4())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for calibration files; install the arm dependency group"
        ) from exc
    return yaml


def load_calibration(path: str | os.PathLike[str]) -> Calibration:
    yaml = _yaml()
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"could not load calibration: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CalibrationError("calibration YAML root must be a mapping")
    return Calibration.from_dict(value)


def save_calibration_atomic(calibration: Calibration, path: str | os.PathLike[str]) -> Calibration:
    """Validate, hash, and atomically replace a calibration YAML file."""

    yaml = _yaml()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    finalized = calibration.with_computed_hash()
    finalized.validate_structure()
    serialized = yaml.safe_dump(finalized.to_dict(), sort_keys=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise CalibrationError(f"could not save calibration: {exc}") from exc
    return finalized
