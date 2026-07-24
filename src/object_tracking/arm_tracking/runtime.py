from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .calibration import Calibration, load_calibration
from .arm_commissioning import load_home_profile
from .depth import DepthFrame, estimate_adaptive_roi_depth
from .geometry import (
    deproject_depth_samples,
    deproject_pixel,
    extract_support_plane,
    generate_pregrasp_target,
    has_support_clearance,
    SupportRegion,
)
from .ik_solver import G1RightArmIK, IKResult, IKUnavailable, default_urdf_path
from .tracking import PositionVelocityFilter
from .visualization import visualization_state


StatusCallback = Callable[[dict[str, Any], bytes | None], None]
SnapshotCallback = Callable[[], dict[str, Any]]
_SUPPORT_PLANE_GRACE_S = 5.000
_SUPPORT_PLANE_ARMED_TTL_S = 30.000


class TrackingTransport(Protocol):
    """ROS-facing transport used by the GB10 tracking runtime.

    The protocol deliberately contains no ROS types so perception and safety
    behavior remain unit-testable without a ROS installation.
    """

    def start(self) -> None: ...

    def close(self) -> None: ...

    def receive_depth(self, timeout_s: float) -> DepthFrame | None: ...

    def arm_state(self) -> dict[str, Any]: ...

    def enable_arm(self, session_id: str, calibration_id: str) -> dict[str, Any]: ...

    def publish_target(
        self,
        session_id: str,
        sequence: int,
        calibration_id: str,
        right_arm_q: Sequence[float],
        pipeline_age_ms: float,
    ) -> None: ...

    def stop_arm(self, reason: str) -> None: ...

    def commissioning(self, command: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RuntimeConfig:
    calibration_path: Path
    tabletop_path: Path | None = None
    arm_home_path: Path | None = None
    robot_id: str | None = None
    execute: bool = False
    target_hz: float = 20.0
    max_pair_skew_s: float = 0.100
    prediction_horizon_s: float = 0.150
    trajectory_model_path: Path | None = None

    def __post_init__(self) -> None:
        if not 10.0 <= self.target_hz <= 30.0:
            raise ValueError("target_hz must be between 10 and 30 Hz")


def register_depth_in_rgb(
    z16: np.ndarray,
    calibration: Calibration,
) -> np.ndarray:
    """Project raw depth pixels into the native RGB profile with a z-buffer."""

    depth = np.asarray(z16, dtype=np.uint16)
    expected = (calibration.depth_profile.height, calibration.depth_profile.width)
    if depth.shape != expected:
        raise ValueError(f"depth frame shape {depth.shape} does not match {expected}")
    points_depth = deproject_depth_samples(
        depth,
        calibration.depth_intrinsics,
        depth_scale=calibration.depth_scale,
        stride=1,
    )
    points_rgb = calibration.depth_to_rgb.apply(points_depth)
    in_front = points_rgb[:, 2] > 0.0
    points_rgb = points_rgb[in_front]
    source_units = np.rint(points_rgb[:, 2] / calibration.depth_scale).astype(np.uint16)
    rgb_intrinsics = calibration.rgb_intrinsics
    u = np.rint(
        points_rgb[:, 0] / points_rgb[:, 2] * rgb_intrinsics.fx + rgb_intrinsics.ppx
    ).astype(int)
    v = np.rint(
        points_rgb[:, 1] / points_rgb[:, 2] * rgb_intrinsics.fy + rgb_intrinsics.ppy
    ).astype(int)
    inside = (
        (u >= 0)
        & (v >= 0)
        & (u < calibration.rgb_profile.width)
        & (v < calibration.rgb_profile.height)
    )
    u, v, source_units = u[inside], v[inside], source_units[inside]
    aligned = np.zeros(
        (calibration.rgb_profile.height, calibration.rgb_profile.width), dtype=np.uint16
    )
    if len(source_units):
        flat_indices = v * calibration.rgb_profile.width + u
        order = np.argsort(source_units)[::-1]
        aligned.reshape(-1)[flat_indices[order]] = source_units[order]
    return aligned


def clamp_point_height_to_support(
    point_xyz: Sequence[float],
    support_plane: Any,
    *,
    minimum_height_m: float = 0.03,
    maximum_height_m: float = 0.12,
) -> tuple[np.ndarray, float]:
    """Keep registered XYZ while bounding only its tabletop-normal height."""

    point = np.asarray(point_xyz, dtype=float)
    height = float(support_plane.signed_distance(point))
    clamped_height = float(np.clip(height, minimum_height_m, maximum_height_m))
    normal = (
        support_plane.plane.normal
        if isinstance(support_plane, SupportRegion)
        else support_plane.normal
    )
    corrected = point + (clamped_height - height) * np.asarray(
        normal, dtype=float
    )
    return corrected, clamped_height


def right_arm_ik_seed(
    arm_state: dict[str, Any],
    home_q: Sequence[float] | None,
) -> list[float]:
    """Prefer the commanded arm, then the observed 29-DOF right-arm state."""

    commanded = arm_state.get("commanded_arm_q")
    if isinstance(commanded, list) and len(commanded) == 14:
        return [float(value) for value in commanded[-7:]]
    visualization = arm_state.get("visualization") or {}
    measured = visualization.get("measured_pose_rad")
    if isinstance(measured, list) and len(measured) == 29:
        return [float(value) for value in measured[22:29]]
    return [float(value) for value in (home_q or (0.0,) * 7)]


def select_start_escape_waypoint(
    measured_q_rad: Sequence[float],
    path: Sequence[Sequence[float]],
    target_index: int,
    *,
    reached_tolerance_rad: float = 0.015,
    tracking_tolerance_rad: float = 0.015,
) -> tuple[tuple[float, ...] | None, int, str | None]:
    """Select one bounded escape waypoint using measured, not commanded, pose."""

    measured = np.asarray(measured_q_rad, dtype=float)
    knots = tuple(np.asarray(knot, dtype=float) for knot in path)
    if (
        measured.shape != (7,)
        or not np.all(np.isfinite(measured))
        or len(knots) < 2
        or any(knot.shape != (7,) or not np.all(np.isfinite(knot)) for knot in knots)
        or not 1 <= target_index < len(knots)
    ):
        return None, target_index, "invalid_escape_path"
    index = target_index
    while (
        index < len(knots)
        and float(np.max(np.abs(measured - knots[index]))) <= reached_tolerance_rad
    ):
        index += 1
    if index >= len(knots):
        return None, index, None
    previous = knots[index - 1]
    target = knots[index]
    # Joint servos do not advance at identical rates. Accept asynchronous
    # progress anywhere inside the component-wise box swept by this already
    # validated edge, plus a small encoder/controller tolerance. Requiring the
    # complete pose to remain near the previous knot falsely rejects a wrist
    # that reaches its target before the shoulder.
    corridor_minimum = np.minimum(previous, target) - tracking_tolerance_rad
    corridor_maximum = np.maximum(previous, target) + tracking_tolerance_rad
    if np.any(measured < corridor_minimum) or np.any(measured > corridor_maximum):
        return None, index, "escape_path_tracking_error"
    if float(np.max(np.abs(target - measured))) > 0.05:
        return None, index, "escape_waypoint_step_too_large"
    return tuple(float(value) for value in target), index, None


def depth_colormap_jpeg(z16: np.ndarray, depth_scale: float) -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None
    depth_m = np.asarray(z16, dtype=np.float32) * depth_scale
    valid = (depth_m >= 0.12) & (depth_m <= 4.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip((4.0 - depth_m[valid]) / 3.88 * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    ok, encoded = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return encoded.tobytes() if ok else None


class ArmTrackingRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        snapshot: SnapshotCallback,
        update_status: StatusCallback,
        *,
        transport: TrackingTransport,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.snapshot = snapshot
        self.update_status = update_status
        self.transport = transport
        self.repo_root = repo_root
        self.calibration = load_calibration(config.calibration_path)
        self.tabletop_corners_px: tuple[tuple[float, float], ...] | None = None
        self.tabletop_size_m: tuple[float, float] | None = None
        self.tabletop_footprint_error: str | None = None
        if config.tabletop_path is not None:
            try:
                value = json.loads(config.tabletop_path.read_text(encoding="utf-8"))
                frame = value["camera_frame"]
                expected_size = (
                    self.calibration.rgb_profile.width,
                    self.calibration.rgb_profile.height,
                )
                if (int(frame["width"]), int(frame["height"])) != expected_size:
                    raise ValueError("tabletop corners use a different RGB profile")
                corners = value["corners_px"]
                expected_names = ("near_left", "near_right", "far_right", "far_left")
                if tuple(str(item["name"]) for item in corners) != expected_names:
                    raise ValueError("tabletop corners are not in the required order")
                parsed = tuple((float(item["x"]), float(item["y"])) for item in corners)
                if len(parsed) != 4 or not np.all(np.isfinite(parsed)):
                    raise ValueError("tabletop footprint must contain four finite corners")
                self.tabletop_corners_px = parsed
                tabletop = value["tabletop"]
                depth_m = float(tabletop["depth_m"])
                width_m = float(tabletop["width_m"])
                if not 0.1 <= depth_m <= 2.0 or not 0.1 <= width_m <= 2.0:
                    raise ValueError("tabletop physical dimensions are invalid")
                self.tabletop_size_m = (depth_m, width_m)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.tabletop_footprint_error = f"{type(exc).__name__}: {exc}"
        if config.execute:
            self.calibration.validate_for_execution(
                camera_serial=self.calibration.camera_serial,
                camera_firmware=self.calibration.camera_firmware,
                rgb_profile=self.calibration.rgb_profile,
                depth_profile=self.calibration.depth_profile,
                waist_rad=self.calibration.waist_reference_rad,
                calibration_id=self.calibration.calibration_id,
            )
        self.filter = PositionVelocityFilter()
        self.learned_forecaster = None
        self.trajectory_model_error: str | None = None
        if config.trajectory_model_path is not None:
            try:
                from object_tracking.trajectory_forecaster import LearnedTrajectoryForecaster

                self.learned_forecaster = LearnedTrajectoryForecaster(config.trajectory_model_path)
            except Exception as exc:
                # A learned forecast is an optimization, never a reason to disable the
                # validated depth/filter safety path.
                self.trajectory_model_error = f"{type(exc).__name__}: {exc}"
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_depth: DepthFrame | None = None
        self.last_sequence = -1
        self.target_sequence = 0
        self.last_target_track: int | None = None
        self.last_process_at = 0.0
        # A support plane is a live safety input, not a calibration constant.
        # Keep the most recent valid extraction so one noisy RANSAC frame does
        # not release an otherwise healthy tracking session. The fallback is
        # bounded to _SUPPORT_PLANE_GRACE_S and therefore still fails closed.
        self.last_support_plane: Any | None = None
        self.last_support_plane_error: str | None = None
        self.last_support_plane_at = 0.0
        self.last_arm_poll_at = 0.0
        self.last_arm_state: dict[str, Any] = {"state": "unreachable"}
        self._pending_predictions: deque[tuple[float, str, np.ndarray]] = deque()
        self._prediction_errors: dict[str, deque[float]] = {
            "learned_trajectory": deque(maxlen=120),
            "alpha_beta_fallback": deque(maxlen=120),
        }
        self._last_prediction_error: dict[str, float] = {}
        self.reference_body_q: tuple[float, ...] | None = None
        self.home_q: tuple[float, ...] | None = None
        if config.arm_home_path is not None:
            profile = load_home_profile(
                config.arm_home_path,
                expected_robot_id=config.robot_id,
            )
            self.home_q = tuple(float(value) for value in profile["measured_q"])
        self.ik: G1RightArmIK | None = None
        self.ik_error: str | None = None
        self._start_escape_path: tuple[tuple[float, ...], ...] | None = None
        self._start_escape_target_index = 1
        try:
            self.ik = G1RightArmIK(default_urdf_path(repo_root))
        except IKUnavailable as exc:
            self.ik_error = str(exc)

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="arm-tracking")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self._stop_arm("runtime_stopped")
        self.transport.close()

    def _run(self) -> None:
        try:
            self.transport.start()
        except Exception as exc:
            self._report(status="error", reason=f"tracking_transport_start_failed: {exc}")
            try:
                self.transport.close()
            except Exception:
                pass
            return
        retry_s = 0.25
        while not self.stop_event.is_set():
            try:
                # A missed ROS depth frame should be detected promptly.  The
                # caller preserves the last safe arm target while the next
                # 30 Hz frame is awaited.
                frame = self.transport.receive_depth(timeout_s=0.20)
                if frame is None:
                    self.last_sequence = -1
                    self._stop_arm("depth_transport_lost")
                    self._report(status="waiting_for_depth", reason="depth_receive_timeout")
                    continue
                retry_s = 0.25
                if frame.sequence <= self.last_sequence:
                    continue
                self.last_sequence = frame.sequence
                self.last_depth = frame
                self._process_latest()
            except Exception as exc:
                self._stop_arm("depth_transport_lost")
                self.last_sequence = -1
                self._report(
                    status="waiting_for_depth",
                    reason=f"tracking_transport_error: {type(exc).__name__}: {exc}",
                )
                self.stop_event.wait(retry_s)
                retry_s = min(retry_s * 2.0, 2.0)

    def _process_latest(self) -> None:
        frame = self.last_depth
        if frame is None:
            return
        now = time.monotonic()
        if now - self.last_process_at < 0.75 / self.config.target_hz:
            return
        self.last_process_at = now
        snapshot = self.snapshot()
        rgb_time = float(snapshot.get("rgb_receipt_time_s", 0.0))
        skew_s = abs(frame.receipt_time_s - rgb_time)
        age_ms = max(0.0, (time.monotonic() - frame.receipt_time_s) * 1000.0)
        if frame.registered_to_rgb:
            expected_rgb_shape = (
                self.calibration.rgb_profile.height,
                self.calibration.rgb_profile.width,
            )
            if frame.z16.shape != expected_rgb_shape:
                self._reject(
                    {
                        "enabled": True,
                        "mode": "execute" if self.config.execute else "dry-run",
                        "depth_sequence": frame.sequence,
                    },
                    "registered_depth_profile_mismatch",
                    None,
                )
                return
            aligned = frame.z16
        else:
            aligned = register_depth_in_rgb(frame.z16, self.calibration)
        colormap = depth_colormap_jpeg(aligned, frame.depth_scale)
        base_status: dict[str, Any] = {
            "enabled": True,
            "mode": "execute" if self.config.execute else "dry-run",
            "depth_sequence": frame.sequence,
            "depth_age_ms": round(age_ms, 3),
            "pair_skew_ms": round(skew_s * 1000.0, 3),
            "calibration_id": self.calibration.calibration_id,
            "target_sequence": self.target_sequence,
            "target_age_ms": round(max(0.0, (time.monotonic() - rgb_time) * 1000.0), 3),
        }
        arm_state = self._arm_state()
        base_status["visualization"] = self._visualization_context(arm_state)
        recognized_ids = {
            self.calibration.calibration_id,
            f"factory-{self.calibration.camera_serial}-{frame.z16.shape[1]}x{frame.z16.shape[0]}",
            f"ros-aligned-{frame.z16.shape[1]}x{frame.z16.shape[0]}",
        }
        frame_calibration_id = frame.calibration_id
        frame_geometry = f"{frame.z16.shape[1]}x{frame.z16.shape[0]}"
        factory_geometry_match = (
            frame_calibration_id.startswith("factory-")
            and frame_calibration_id.endswith(f"-{frame_geometry}")
        )
        if (self.config.execute and frame.calibration_id != self.calibration.calibration_id) or (
            not self.config.execute and frame_calibration_id not in recognized_ids and not factory_geometry_match
        ):
            self._reject(base_status, "calibration_mismatch", colormap)
            return
        if skew_s > self.config.max_pair_skew_s:
            self._reject(base_status, "rgb_depth_pair_stale", colormap)
            return

        tracks = snapshot.get("tracks") or []
        eligible = [
            item
            for item in tracks
            if item.get("bbox_xyxy")
            and item.get("track_id") is not None
            and float(item.get("confidence", 0.0)) >= 0.25
        ]
        selected = next(
            (item for item in eligible if int(item["track_id"]) == self.last_target_track),
            None,
        )
        if selected is None and eligible:
            selected = max(eligible, key=lambda item: float(item.get("confidence", 0.0)))
        if selected is None:
            self.filter.reset()
            self._reject(base_status, "target_lost", colormap)
            return
        base_status["detector_confidence"] = round(float(selected.get("confidence", 0.0)), 5)
        base_status["depth_valid"] = False
        rgb_shape = snapshot.get("rgb_shape")
        if not rgb_shape:
            self._reject(base_status, "rgb_unavailable", colormap)
            return
        rgb_height, rgb_width = (int(value) for value in rgb_shape[:2])
        sx = self.calibration.rgb_profile.width / rgb_width
        sy = self.calibration.rgb_profile.height / rgb_height
        bbox = (
            float(selected["bbox_xyxy"][0]) * sx,
            float(selected["bbox_xyxy"][1]) * sy,
            float(selected["bbox_xyxy"][2]) * sx,
            float(selected["bbox_xyxy"][3]) * sy,
        )
        plane = self._support_plane(
            aligned,
            frame.depth_scale,
            freeze=arm_state.get("state") in ("ARMING", "ARMED"),
        )
        if plane is not None:
            base_status["visualization"]["support_plane"] = plane.to_dict()
        base_status["visualization"]["support_plane_status"] = {
            "available": plane is not None,
            "age_ms": round(max(0.0, time.monotonic() - self.last_support_plane_at) * 1000.0, 1)
            if self.last_support_plane_at
            else None,
            "error": self.last_support_plane_error,
        }
        if plane is None:
            self._reject(base_status, "support_plane_unavailable", colormap)
            return
        estimate = estimate_adaptive_roi_depth(
            aligned,
            bbox,
            depth_scale=frame.depth_scale,
        )
        if estimate is None or not estimate.is_certain:
            self._reject(base_status, "depth_uncertain", colormap)
            return
        # Use the registered depth point for the actual object XYZ.  The plane is
        # retained as a bounded tabletop/clearance validator; intersecting a ray
        # with a stale plane can otherwise extrapolate to metres outside the robot.
        optical_depth_point = deproject_pixel(
            estimate.pixel_xy,
            estimate.depth_m,
            self.calibration.rgb_intrinsics,
        )
        direct_torso = self.calibration.optical_to_torso.apply(optical_depth_point)
        if not np.all(np.isfinite(direct_torso)) or np.linalg.norm(direct_torso) > 2.0:
            self._reject(base_status, "localization_outlier", colormap)
            return
        torso, object_height_m = clamp_point_height_to_support(direct_torso, plane)
        base_status.update(
            {
                "depth_valid": True,
                "median_aligned_depth_m": round(float(estimate.depth_m), 5),
                "object_depth_certain": True,
                "localization_source": "registered_depth_table_height_clamped",
                "object_height_above_table_m": round(object_height_m, 5),
            }
        )
        if self.filter.state is not None and rgb_time <= self.filter.state.timestamp_s:
            self._reject(base_status, "waiting_for_new_rgb_frame", colormap)
            return
        tracked = self.filter.update(torso, rgb_time)
        alpha_beta_prediction = tracked.predict(self.config.prediction_horizon_s)
        learned_prediction = None
        if self.learned_forecaster is not None:
            self.learned_forecaster.update(torso, rgb_time)
            learned_prediction = self.learned_forecaster.predict(self.config.prediction_horizon_s)
        self._score_due_predictions(torso, rgb_time)
        due_time = rgb_time + self.config.prediction_horizon_s
        self._queue_prediction("alpha_beta_fallback", alpha_beta_prediction, due_time)
        if learned_prediction is not None:
            self._queue_prediction("learned_trajectory", learned_prediction, due_time)
        predicted = learned_prediction if learned_prediction is not None else alpha_beta_prediction
        # Stop at the bunny's near surface rather than applying the generic
        # 20 cm manipulation stand-off.  The measured plush radius (~5.5 cm)
        # plus half the palm thickness (~1.2 cm) calls for a 7 cm preview
        # offset, which remains non-contacting and inside the certified G1
        # workspace. Swept-link/table checks still gate the resulting IK path.
        target = generate_pregrasp_target(
            predicted,
            shoulder_position=(0.0, -0.18, 0.35),
            stand_off_m=0.07,
        )
        base_status.update(
            {
                "status": "tracking",
                "track_id": int(selected["track_id"]),
                "depth_valid_fraction": (
                    None if estimate is None else round(estimate.valid_fraction, 4)
                ),
                "object_xyz_m": torso.round(5).tolist(),
                "predicted_xyz_m": predicted.round(5).tolist(),
                "prediction_source": (
                    "learned_trajectory"
                    if learned_prediction is not None
                    else "alpha_beta_fallback"
                ),
                "trajectory_model": (
                    None
                    if self.config.trajectory_model_path is None
                    else str(self.config.trajectory_model_path)
                ),
                "trajectory_model_error": self.trajectory_model_error,
                "alpha_beta_predicted_xyz_m": alpha_beta_prediction.round(5).tolist(),
                "learned_predicted_xyz_m": (
                    None if learned_prediction is None else learned_prediction.round(5).tolist()
                ),
                "prediction_evaluation": self._prediction_evaluation(),
                "target_xyz_m": target.position.round(5).tolist(),
            }
        )
        base_status["visualization"].update(
            {
                "measured_object_xyz_m": base_status["object_xyz_m"],
                "predicted_object_xyz_m": base_status["predicted_xyz_m"],
                "predicted_trajectory_xyz_m": [
                    base_status["object_xyz_m"],
                    base_status["predicted_xyz_m"],
                ],
                "pregrasp_target_xyz_m": base_status["target_xyz_m"],
            }
        )
        if self.ik is None:
            base_status.update({"ik_status": "unavailable", "ik_error": self.ik_error})
            self._reject(base_status, "ik_unavailable", colormap)
            return
        last_q = right_arm_ik_seed(arm_state, self.home_q)
        collision_labels = self.ik.collision_labels(last_q)
        escape_needed = self._start_escape_path is not None or bool(collision_labels)
        # The hip-rest escape is an independently planned and fully
        # link/table-validated path. A noisy bunny endpoint must not interrupt
        # it; endpoint workspace/clearance gates become relevant only when
        # task-directed Cartesian IK is about to begin.
        if not escape_needed:
            if not self.calibration.workspace.contains(target.position):
                self._reject(base_status, "workspace_violation", colormap)
                return
            if not has_support_clearance(target.position, plane):
                self._reject(base_status, "support_plane_clearance", colormap)
                return
        ik_step_type = "analytic_local_translation"
        # Once a validated hip-rest escape starts, latch it until measured
        # state reaches the collision-free endpoint. The contact sits at the
        # mesh boundary and can flicker clear for one encoder sample; dropping
        # the path on that sample would incorrectly attempt task IK from the
        # factory rest pose.
        if escape_needed:
            ik_step_type = "guided_table_clearance"
            if self._start_escape_path is None:
                escape = self.ik.plan_guided_clearance(
                    last_q,
                    support_plane=plane,
                    lift_m=0.25,
                    forward_m=0.06,
                )
                if not escape.ok or escape.q_path is None:
                    base_status.update(
                        {
                            "ik_status": escape.reason,
                            "ik_step_type": ik_step_type,
                            "ik_collision_labels": list(collision_labels),
                            "ik_global_backend": self.ik.global_backend,
                            "ik_local_backend": self.ik.local_backend,
                            "arm_state": arm_state.get("state", "dry-run"),
                        }
                    )
                    self._reject(
                        base_status,
                        f"ik_{escape.reason or 'start_collision_escape_failed'}",
                        colormap,
                    )
                    return
                self._start_escape_path = escape.q_path
                self._start_escape_target_index = 1
            waypoint, waypoint_index, selection_error = select_start_escape_waypoint(
                last_q,
                self._start_escape_path,
                self._start_escape_target_index,
            )
            self._start_escape_target_index = waypoint_index
            if selection_error is not None:
                self._start_escape_path = None
                self._reject(base_status, f"ik_{selection_error}", colormap)
                return
            if waypoint is None:
                self._start_escape_path = None
                collision_labels = self.ik.collision_labels(last_q)
                if collision_labels:
                    self._reject(
                        base_status,
                        "ik_escape_complete_but_collision_remains",
                        colormap,
                    )
                    return
                if not self.calibration.workspace.contains(target.position):
                    self._reject(base_status, "workspace_violation", colormap)
                    return
                if not has_support_clearance(target.position, plane):
                    self._reject(base_status, "support_plane_clearance", colormap)
                    return
                ik_step_type = "analytic_local_translation"
                transform = self.ik.forward_kinematics(last_q)
                transform[:3, 3] = target.position
                ik = self.ik.solve_local_translation(
                    transform,
                    last_q,
                    support_plane=plane,
                )
            else:
                # The complete cached path was densely collision/table
                # validated when planned. Expose one waypoint repeatedly until
                # measured state reaches it; the selector rejects motion
                # outside the validated joint-wise corridor and the bridge
                # independently rejects >0.05-rad targets.
                ik = IKResult(True, waypoint, 0.0, 0.0)
        else:
            self._start_escape_path = None
            self._start_escape_target_index = 1
            # Realtime tracking is an incremental Cartesian servo. Preserve
            # measured wrist orientation and expose one fully validated edge.
            transform = self.ik.forward_kinematics(last_q)
            transform[:3, 3] = target.position
            ik = self.ik.solve_local_translation(
                transform,
                last_q,
                support_plane=plane,
            )
        base_status.update(
            {
                "ik_status": "ok" if ik.ok else ik.reason,
                "ik_step_type": ik_step_type,
                "ik_collision_labels": list(collision_labels),
                "ik_escape_waypoint": (
                    self._start_escape_target_index
                    if ik_step_type == "guided_table_clearance"
                    else None
                ),
                "ik_escape_waypoint_count": (
                    len(self._start_escape_path)
                    if self._start_escape_path is not None
                    else None
                ),
                "ik_global_backend": self.ik.global_backend,
                "ik_local_backend": self.ik.local_backend,
                "ik_position_error_m": ik.position_error_m,
                "ik_orientation_error_rad": ik.orientation_error_rad,
                "arm_state": arm_state.get("state", "dry-run"),
                "arm_weight": arm_state.get("weight"),
                "arm_last_target_age_ms": arm_state.get("last_target_age_ms"),
                "arm_loop": arm_state.get("loop") or {},
            }
        )
        if not ik.ok or ik.q_rad is None:
            self._reject(base_status, f"ik_{ik.reason}", colormap)
            return
        base_status["predicted_bounded_arm_command_rad"] = [
            round(float(value), 6) for value in ik.q_rad
        ]
        base_status["processing_latency_ms"] = round(
            max(0.0, (time.monotonic() - rgb_time) * 1000.0), 3
        )
        robot_visual = arm_state.get("visualization") or {}
        measured_body = robot_visual.get("measured_pose_rad")
        if isinstance(measured_body, list) and len(measured_body) == 29:
            left_arm = measured_body[15:22]
            commanded_arm = [*left_arm, *[float(value) for value in ik.q_rad]]
            base_status["visualization"] = {
                **base_status["visualization"],
                **visualization_state(
                    measured_body_q=measured_body,
                    measured_body_dq=robot_visual.get("measured_velocity_rad_s"),
                    commanded_arm_q=commanded_arm,
                    reference_body_q=self.reference_body_q,
                    received_at=robot_visual.get("received_monotonic_s"),
                    now=(
                        None
                        if robot_visual.get("received_monotonic_s") is None
                        else float(robot_visual["received_monotonic_s"])
                        + float(robot_visual.get("state_age_ms") or 0.0) / 1000.0
                    ),
                    state_ttl_s=0.25,
                    available=bool(robot_visual.get("available")),
                    faulted_joints=robot_visual.get("faulted_joints") or (),
                ),
                "workspace_bounds": base_status["visualization"]["workspace_bounds"],
                "camera": base_status["visualization"]["camera"],
                "support_plane": base_status["visualization"].get("support_plane"),
                "measured_object_xyz_m": base_status.get("object_xyz_m"),
                "predicted_object_xyz_m": base_status.get("predicted_xyz_m"),
                "predicted_trajectory_xyz_m": [
                    base_status.get("object_xyz_m"),
                    base_status.get("predicted_xyz_m"),
                ],
                "pregrasp_target_xyz_m": base_status.get("target_xyz_m"),
                "end_effector_xyz_m": base_status.get("target_xyz_m"),
                "ik_result": {
                    "ok": True,
                    "right_arm_q_rad": [float(value) for value in ik.q_rad],
                    "position_error_m": ik.position_error_m,
                    "orientation_error_rad": ik.orientation_error_rad,
                },
            }
        if self.config.execute:
            # Stream bounded targets during the controller's weight ramp.  If
            # we wait for ARMED, the target deadman expires during ARMING and
            # the bridge releases immediately after a small twitch.
            if arm_state.get("state") not in ("ARMING", "ARMED") or not arm_state.get(
                "session_id"
            ):
                self._reject(base_status, "arm_not_explicitly_enabled", colormap)
                return
            pipeline_age_ms = max(
                0.0,
                (time.monotonic() - min(frame.receipt_time_s, rgb_time)) * 1000.0,
            )
            try:
                self.transport.publish_target(
                    session_id=str(arm_state["session_id"]),
                    sequence=self.target_sequence,
                    calibration_id=self.calibration.calibration_id,
                    right_arm_q=[float(value) for value in ik.q_rad],
                    pipeline_age_ms=pipeline_age_ms,
                )
            except Exception as exc:
                base_status["transport_error"] = f"{type(exc).__name__}: {exc}"
                self._stop_arm("arm_target_publish_failed", force=True)
                self._reject(base_status, "arm_target_publish_failed", colormap)
                return
            self.target_sequence += 1
            base_status["status"] = "target_sent"
            base_status["target_sequence"] = self.target_sequence
            base_status["pipeline_age_ms"] = round(pipeline_age_ms, 3)
        self.last_target_track = int(selected["track_id"])
        self.update_status(base_status, colormap)

    def _queue_prediction(self, source: str, position: np.ndarray, due_time_s: float) -> None:
        """Keep short-lived forecasts until a future measured position can score them."""

        self._pending_predictions.append((due_time_s, source, position.copy()))
        while len(self._pending_predictions) > 300:
            self._pending_predictions.popleft()

    def _score_due_predictions(self, actual_position: np.ndarray, timestamp_s: float) -> None:
        while self._pending_predictions and self._pending_predictions[0][0] <= timestamp_s:
            _, source, prediction = self._pending_predictions.popleft()
            error_m = float(np.linalg.norm(prediction - actual_position))
            if np.isfinite(error_m) and source in self._prediction_errors:
                self._prediction_errors[source].append(error_m)
                self._last_prediction_error[source] = error_m

    def _prediction_evaluation(self) -> dict[str, Any]:
        """A small online holdout: forecast now, compare against later 3D observation."""

        methods: dict[str, Any] = {}
        for source, errors in self._prediction_errors.items():
            methods[source] = {
                "samples": len(errors),
                "last_error_m": self._last_prediction_error.get(source),
                "mean_error_m": (None if not errors else round(float(np.mean(errors)), 5)),
                "median_error_m": (None if not errors else round(float(np.median(errors)), 5)),
            }
        learned = methods["learned_trajectory"]
        fallback = methods["alpha_beta_fallback"]
        verdict = "collecting"
        if learned["samples"] >= 10 and fallback["samples"] >= 10:
            verdict = (
                "learned_better"
                if learned["mean_error_m"] < fallback["mean_error_m"]
                else "fallback_better_or_equal"
            )
        return {
            "horizon_ms": round(self.config.prediction_horizon_s * 1000.0, 1),
            "verdict": verdict,
            "methods": methods,
        }

    def _support_plane(
        self,
        aligned: np.ndarray,
        depth_scale: float,
        *,
        freeze: bool = False,
    ) -> SupportRegion | None:
        if (
            freeze
            and self.last_support_plane is not None
            and time.monotonic() - self.last_support_plane_at
            <= _SUPPORT_PLANE_ARMED_TTL_S
        ):
            return self.last_support_plane
        try:
            if self.tabletop_corners_px is None:
                raise ValueError(
                    "no calibrated tabletop footprint"
                    + (
                        ""
                        if self.tabletop_footprint_error is None
                        else f": {self.tabletop_footprint_error}"
                    )
                )
            plane, _ = extract_support_plane(
                aligned,
                self.calibration.rgb_intrinsics,
                depth_scale=depth_scale,
                optical_to_base=self.calibration.optical_to_torso,
                pixel_roi=self.tabletop_corners_px,
                # The polygon encloses the tabletop in RGB pixels, but its far
                # edge can still include background/floor depth through gaps.
                # Restrict RANSAC candidates to the calibrated tabletop-height
                # corridor before choosing the dominant plane.
                base_minimum=(
                    float(self.calibration.workspace.minimum[0] - 0.12),
                    float(self.calibration.workspace.minimum[1] - 0.12),
                    self._expected_table_height_m() - self._table_height_tolerance_m(),
                ),
                base_maximum=(
                    float(self.calibration.workspace.maximum[0] + 0.12),
                    float(self.calibration.workspace.maximum[1] + 0.12),
                    self._expected_table_height_m() + self._table_height_tolerance_m(),
                ),
                stride=8,
            )
            origin = self.calibration.optical_to_torso.translation
            rotation = self.calibration.optical_to_torso.rotation
            intrinsics = self.calibration.rgb_intrinsics
            corners_xyz: list[np.ndarray] = []
            for u, v in self.tabletop_corners_px:
                optical_ray = np.asarray(
                    (
                        (u - intrinsics.ppx) / intrinsics.fx,
                        (v - intrinsics.ppy) / intrinsics.fy,
                        1.0,
                    ),
                    dtype=float,
                )
                torso_ray = rotation @ optical_ray
                denominator = float(np.dot(plane.normal, torso_ray))
                if abs(denominator) <= 1e-6:
                    raise ValueError("tabletop corner ray is parallel to the support plane")
                distance = -float(np.dot(plane.normal, origin) + plane.offset) / denominator
                if distance <= 0.0:
                    raise ValueError("tabletop corner intersects behind the camera")
                corners_xyz.append(origin + distance * torso_ray)
            support = SupportRegion.from_ordered_corners(
                plane,
                corners_xyz,
                certified_edges=("u_min", "u_max", "v_min", "v_max"),
                lateral_margin_m=0.07,
                source="live_plane_calibrated_pixel_corners",
            )
            if self.tabletop_size_m is None:
                raise ValueError("no calibrated tabletop physical dimensions")
            measured_size = support.maximum_uv - support.minimum_uv
            expected_size = np.asarray(self.tabletop_size_m, dtype=float)
            size_ratio = measured_size / expected_size
            if np.any(size_ratio < 0.65) or np.any(size_ratio > 1.35):
                raise ValueError(
                    "live tabletop footprint dimensions disagree with calibration: "
                    f"measured={measured_size.round(3).tolist()}m "
                    f"expected={expected_size.round(3).tolist()}m"
                )
            self.last_support_plane = support
            self.last_support_plane_at = time.monotonic()
            self.last_support_plane_error = None
            return support
        except ValueError as exc:
            self.last_support_plane_error = f"{type(exc).__name__}: {exc}"
            if (
                self.last_support_plane is not None
                and time.monotonic() - self.last_support_plane_at
                <= _SUPPORT_PLANE_GRACE_S
            ):
                return self.last_support_plane
            return None

    def _expected_table_height_m(self) -> float:
        poses = (*self.calibration.solve_poses, *self.calibration.validation_poses)
        return float(np.median([pose.torso_point_m[2] for pose in poses]))

    def _table_height_tolerance_m(self) -> float:
        residual = max(
            self.calibration.solve_residuals.max_error_m,
            self.calibration.validation_residuals.max_error_m,
        )
        return max(0.08, float(residual) + 0.04)

    def _arm_state(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self.last_arm_poll_at < 0.1:
            return self.last_arm_state
        self.last_arm_poll_at = now
        try:
            self.last_arm_state = dict(self.transport.arm_state())
        except Exception:
            self.last_arm_state = {"state": "unreachable"}
        return self.last_arm_state

    def _visualization_context(self, arm_state: dict[str, Any]) -> dict[str, Any]:
        robot_visual = arm_state.get("visualization")
        if not isinstance(robot_visual, dict):
            robot_visual = visualization_state(
                measured_body_q=None,
                available=False,
                state_ttl_s=0.25,
            )
        measured = robot_visual.get("measured_pose_rad")
        if self.reference_body_q is None and isinstance(measured, list) and len(measured) == 29:
            reference = list(float(value) for value in measured)
            if self.home_q is not None:
                reference[22:29] = self.home_q
            self.reference_body_q = tuple(reference)
        result = {
            **robot_visual,
            "reference_pose_rad": (
                list(self.reference_body_q)
                if self.reference_body_q is not None
                else robot_visual.get("reference_pose_rad")
            ),
            "workspace_bounds": {
                "minimum": self.calibration.workspace.minimum.tolist(),
                "maximum": self.calibration.workspace.maximum.tolist(),
            },
            "camera": {
                "intrinsics": self.calibration.rgb_intrinsics.to_dict(),
                "optical_to_torso": self.calibration.optical_to_torso.to_dict(),
            },
            "support_plane": None,
            "support_plane_status": {
                "available": False,
                "age_ms": None,
                "error": self.last_support_plane_error,
            },
            "measured_object_xyz_m": None,
            "predicted_object_xyz_m": None,
            "predicted_trajectory_xyz_m": [],
            "pregrasp_target_xyz_m": None,
            "end_effector_xyz_m": None,
            "ik_result": None,
        }
        return result

    def _stop_arm(self, reason: str, *, force: bool = False) -> None:
        if not self.config.execute or (self.last_target_track is None and not force):
            return
        try:
            self.transport.stop_arm(reason)
        except Exception:
            pass
        self.last_target_track = None

    def _reject(self, status: dict[str, Any], reason: str, colormap: bytes | None) -> None:
        status.update({"status": "rejected", "reason": reason})
        self._stop_arm(reason)
        self.update_status(status, colormap)

    def _report(self, **values: Any) -> None:
        status = {
            "enabled": True,
            "mode": "execute" if self.config.execute else "dry-run",
            "calibration_id": self.calibration.calibration_id,
            **values,
        }
        self.update_status(status, None)
