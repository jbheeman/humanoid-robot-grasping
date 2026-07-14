"""Command-line calibration workflow for probe/collect/solve/validate/tune.

Invoke with ``python -m object_tracking.arm_tracking.calibration_cli``.  The
RealSense SDK is imported only by ``probe``; every other command is replayable
from YAML correspondence files without a camera or robot.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import (
    Calibration,
    CalibrationError,
    CalibrationPose,
    MIN_SOLVE_POSES,
    MIN_VALIDATION_POSES,
    SCHEMA_VERSION,
    load_calibration,
    new_calibration_id,
    save_calibration_atomic,
    solve_rigid_transform,
    tune_transform,
    utc_timestamp,
    validate_correspondences,
    validate_registration_correspondences,
)


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the calibration CLI") from exc
    return yaml


def _read_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    yaml = _yaml()
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{path} must contain a YAML mapping")
    return dict(value)


def _write_mapping_atomic(value: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    yaml = _yaml()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(dict(value), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except (OSError, yaml.YAMLError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise CalibrationError(f"could not write {path}: {exc}") from exc


def probe_realsense() -> dict[str, Any]:
    """Capture the active factory camera metadata needed by a template."""

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required only for calibration probe") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    # The normal robot launcher already owns the RGB V4L relay.  Open only
    # depth here, then query the device's factory color profiles below.  This
    # avoids a second RGB stream request failing on a robot-mounted D435.
    config.enable_stream(rs.stream.depth, rs.format.z16)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        depth_sensor = device.first_depth_sensor()

        color_candidates = []
        for sensor in device.query_sensors():
            for candidate in sensor.get_stream_profiles():
                try:
                    video = candidate.as_video_stream_profile()
                    if video.stream_type() != rs.stream.color:
                        continue
                    intrinsics = video.get_intrinsics()
                    extrinsics = depth_profile.get_extrinsics_to(video)
                    color_candidates.append((video, intrinsics, extrinsics))
                except RuntimeError:
                    continue
        if not color_candidates:
            raise RuntimeError("RealSense device exposes no usable color stream profiles")

        # The lab relay runs 960x540 at 60 FPS. Prefer that exact factory
        # profile, otherwise use the densest available profile rather than
        # failing the probe because a particular RGB format is unavailable.
        rgb_profile, rgb_intrinsics, extrinsics = max(
            color_candidates,
            key=lambda item: (
                item[0].width() == 960 and item[0].height() == 540 and item[0].fps() == 60,
                item[0].width() * item[0].height(),
                item[0].fps(),
            ),
        )

        def stream(value: Any) -> dict[str, Any]:
            return {
                "width": value.width(),
                "height": value.height(),
                "fps": value.fps(),
                "format": str(value.format()).split(".")[-1].lower(),
            }

        def intrinsics(value: Any) -> dict[str, Any]:
            return {
                "width": value.width,
                "height": value.height,
                "fx": value.fx,
                "fy": value.fy,
                "ppx": value.ppx,
                "ppy": value.ppy,
                "distortion_model": str(value.model).split(".")[-1].lower(),
                "coefficients": list(value.coeffs),
            }

        return {
            "schema_version": SCHEMA_VERSION,
            "camera": {
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "rgb_profile": stream(rgb_profile),
                "depth_profile": stream(depth_profile),
                "rgb_intrinsics": intrinsics(rgb_intrinsics),
                "depth_intrinsics": intrinsics(depth_intrinsics),
                "depth_scale": depth_sensor.get_depth_scale(),
                "depth_to_rgb": {
                    "rotation": np.asarray(extrinsics.rotation, dtype=float).reshape(3, 3).tolist(),
                    "translation": list(extrinsics.translation),
                },
            },
        }
    finally:
        pipeline.stop()


def append_correspondence(
    path: str | os.PathLike[str],
    *,
    subset: str,
    name: str,
    optical_point_m: Sequence[float],
    torso_point_m: Sequence[float],
    source: str,
    orientation_error_deg: float | None = None,
) -> CalibrationPose:
    if subset not in {"solve", "validation"}:
        raise CalibrationError("subset must be solve or validation")
    destination = Path(path)
    collection = (
        _read_mapping(destination)
        if destination.exists()
        else {"schema_version": SCHEMA_VERSION, "solve_poses": [], "validation_poses": []}
    )
    if int(collection.get("schema_version", 0)) != SCHEMA_VERSION:
        raise CalibrationError("unsupported correspondence schema")
    all_values = list(collection.get("solve_poses", [])) + list(
        collection.get("validation_poses", [])
    )
    if any(str(value.get("name")) == name for value in all_values):
        raise CalibrationError(f"pose name {name!r} already exists")
    pose = CalibrationPose(
        name=name,
        optical_point_m=tuple(float(item) for item in optical_point_m),
        torso_point_m=tuple(float(item) for item in torso_point_m),
        orientation_error_deg=orientation_error_deg,
        source=source,
    )
    key = f"{subset}_poses"
    values = list(collection.get(key, []))
    values.append(pose.to_dict())
    collection[key] = values
    collection.setdefault("solve_poses", [])
    collection.setdefault("validation_poses", [])
    _write_mapping_atomic(collection, destination)
    return pose


def solve_from_files(
    template_path: str | os.PathLike[str],
    collection_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> Calibration:
    template = _read_mapping(template_path)
    collection = _read_mapping(collection_path)
    solve_poses = tuple(
        CalibrationPose.from_dict(value) for value in collection.get("solve_poses", [])
    )
    validation_poses = tuple(
        CalibrationPose.from_dict(value) for value in collection.get("validation_poses", [])
    )
    if len(solve_poses) < MIN_SOLVE_POSES:
        raise CalibrationError(f"at least {MIN_SOLVE_POSES} solve poses are required")
    if len(validation_poses) < MIN_VALIDATION_POSES:
        raise CalibrationError(
            f"at least {MIN_VALIDATION_POSES} held-out validation poses are required"
        )
    solution = solve_rigid_transform(
        (pose.optical_point_m for pose in solve_poses),
        (pose.torso_point_m for pose in solve_poses),
        orientation_errors_deg=(
            pose.orientation_error_deg
            for pose in solve_poses
            if pose.orientation_error_deg is not None
        ),
    )
    solved_poses = tuple(
        replace(pose, position_error_m=float(error))
        for pose, error in zip(solve_poses, solution.errors_m, strict=True)
    )
    validation_residuals = validate_correspondences(solution.transform, validation_poses)
    validation_errors = np.linalg.norm(
        solution.transform.apply(np.asarray([pose.optical_point_m for pose in validation_poses]))
        - np.asarray([pose.torso_point_m for pose in validation_poses]),
        axis=1,
    )
    validated_poses = tuple(
        replace(pose, position_error_m=float(error))
        for pose, error in zip(validation_poses, validation_errors, strict=True)
    )
    template.update(
        {
            "schema_version": SCHEMA_VERSION,
            "calibration_id": template.get("calibration_id") or new_calibration_id(),
            "calibration_hash": "",
            "created_at": template.get("created_at") or utc_timestamp(),
            "optical_to_torso": solution.transform.to_dict(),
            "solve_poses": [pose.to_dict() for pose in solved_poses],
            "validation_poses": [pose.to_dict() for pose in validated_poses],
            "solve_residuals": solution.residuals.to_dict(),
            "validation_residuals": validation_residuals.to_dict(),
        }
    )
    calibration = Calibration.from_dict(template, check_hash=False).with_computed_hash()
    return save_calibration_atomic(calibration, output_path)


def retune_calibration(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    xyz_delta_m: Sequence[float],
    rpy_delta_deg: Sequence[float],
) -> Calibration:
    calibration = load_calibration(input_path)
    transform = tune_transform(
        calibration.optical_to_torso,
        xyz_delta_m=xyz_delta_m,
        rpy_delta_deg=rpy_delta_deg,
    )
    solve_residuals = validate_correspondences(transform, calibration.solve_poses)
    validation_residuals = validate_correspondences(transform, calibration.validation_poses)

    def annotate(poses: Sequence[CalibrationPose]) -> tuple[CalibrationPose, ...]:
        optical = np.asarray([pose.optical_point_m for pose in poses])
        torso = np.asarray([pose.torso_point_m for pose in poses])
        errors = np.linalg.norm(transform.apply(optical) - torso, axis=1)
        return tuple(
            replace(pose, position_error_m=float(error))
            for pose, error in zip(poses, errors, strict=True)
        )

    tuned = replace(
        calibration,
        calibration_hash="",
        created_at=utc_timestamp(),
        optical_to_torso=transform,
        solve_poses=annotate(calibration.solve_poses),
        validation_poses=annotate(calibration.validation_poses),
        solve_residuals=solve_residuals,
        validation_residuals=validation_residuals,
    ).with_computed_hash()
    return save_calibration_atomic(tuned, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="probe factory RealSense metadata")
    probe.add_argument("--output", type=Path)
    probe.add_argument(
        "--camera-metadata",
        type=Path,
        help="use a previously captured camera metadata mapping instead of live hardware",
    )
    probe.add_argument(
        "--registration-correspondences",
        type=Path,
        help="YAML with expected_rgb_pixels and measured_depth_pixels",
    )

    collect = subparsers.add_parser("collect", help="append one solve/validation correspondence")
    collect.add_argument("collection", type=Path)
    collect.add_argument("--subset", choices=("solve", "validation"), required=True)
    collect.add_argument("--name", required=True)
    collect.add_argument("--optical", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    collect.add_argument("--torso", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    collect.add_argument("--source", choices=("apriltag", "manual"), required=True)
    collect.add_argument("--orientation-error-deg", type=float)

    solve = subparsers.add_parser("solve", help="solve and validate a complete calibration")
    solve.add_argument("template", type=Path)
    solve.add_argument("collection", type=Path)
    solve.add_argument("output", type=Path)

    validate = subparsers.add_parser("validate", help="fail closed on execution calibration gates")
    validate.add_argument("calibration", type=Path)
    validate.add_argument("--camera-serial")
    validate.add_argument("--camera-firmware")
    validate.add_argument("--calibration-id")
    validate.add_argument("--waist-deg", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"))

    tune = subparsers.add_parser("tune", help="apply bounded XYZ/RPY tuning and revalidate")
    tune.add_argument("calibration", type=Path)
    tune.add_argument("output", type=Path)
    tune.add_argument("--xyz", nargs=3, type=float, default=(0, 0, 0), metavar=("X", "Y", "Z"))
    tune.add_argument("--rpy-deg", nargs=3, type=float, default=(0, 0, 0), metavar=("R", "P", "Y"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "probe":
            result = (
                _read_mapping(arguments.camera_metadata)
                if arguments.camera_metadata
                else probe_realsense()
            )
            if arguments.registration_correspondences:
                correspondences = _read_mapping(arguments.registration_correspondences)
                residuals = validate_registration_correspondences(
                    correspondences.get("expected_rgb_pixels", []),
                    correspondences.get("measured_depth_pixels", []),
                )
                result["registration_residuals"] = residuals.to_dict()
            if arguments.output:
                _write_mapping_atomic(result, arguments.output)
            else:
                print(_yaml().safe_dump(result, sort_keys=False), end="")
        elif arguments.command == "collect":
            pose = append_correspondence(
                arguments.collection,
                subset=arguments.subset,
                name=arguments.name,
                optical_point_m=arguments.optical,
                torso_point_m=arguments.torso,
                source=arguments.source,
                orientation_error_deg=arguments.orientation_error_deg,
            )
            print(json.dumps(pose.to_dict(), sort_keys=True))
        elif arguments.command == "solve":
            calibration = solve_from_files(
                arguments.template, arguments.collection, arguments.output
            )
            print(json.dumps(calibration.validation_residuals.to_dict(), sort_keys=True))
        elif arguments.command == "validate":
            calibration = load_calibration(arguments.calibration)
            waist = (
                np.deg2rad(arguments.waist_deg)
                if arguments.waist_deg is not None
                else calibration.waist_reference_rad
            )
            calibration.validate_for_execution(
                camera_serial=arguments.camera_serial or calibration.camera_serial,
                camera_firmware=arguments.camera_firmware,
                rgb_profile=calibration.rgb_profile,
                depth_profile=calibration.depth_profile,
                waist_rad=waist,
                calibration_id=arguments.calibration_id,
            )
            print(json.dumps({"valid": True, "calibration_id": calibration.calibration_id}))
        elif arguments.command == "tune":
            calibration = retune_calibration(
                arguments.calibration,
                arguments.output,
                xyz_delta_m=arguments.xyz,
                rpy_delta_deg=arguments.rpy_deg,
            )
            print(json.dumps(calibration.validation_residuals.to_dict(), sort_keys=True))
    except (CalibrationError, RuntimeError) as exc:
        print(f"calibration {arguments.command} failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
