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
    detect_automatic_support_region,
    deproject_depth_samples,
    deproject_pixel,
    extract_support_plane,
    generate_pregrasp_target,
    has_support_clearance,
    Plane,
    SupportRegion,
    TargetPose,
)
from .ik_solver import G1RightArmIK, IKResult, IKUnavailable, default_urdf_path
from .interception import (
    InterceptDecision,
    InterceptObservation,
    InterceptState,
    LiveInterceptConfig,
    LiveInterceptController,
    load_live_intercept_config,
)
from .tracking import PositionVelocityFilter
from .visualization import visualization_state


StatusCallback = Callable[[dict[str, Any], bytes | None], None]
SnapshotCallback = Callable[[], dict[str, Any]]
_SUPPORT_PLANE_GRACE_S = 5.000
# Plane fitting is CPU-heavy and the tabletop cannot physically change at the
# 20 Hz arm-servo rate. Reuse a recent validated result between 1 Hz refreshes.
_SUPPORT_PLANE_REFRESH_S = 1.000
# The table is localized immediately before arming and remains physically
# fixed during one operator demo. Avoid rerunning expensive plane fitting in
# the realtime path midway through that session.
_SUPPORT_PLANE_ARMED_TTL_S = 300.000
_SOFT_PERCEPTION_REJECTIONS = frozenset(
    {
        "depth_uncertain",
        "rgb_depth_pair_stale",
        "target_lost",
        "waiting_for_new_rgb_frame",
    }
)


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
    allow_nominal_support_plane: bool = False
    automatic_support_plane: bool = True
    automatic_support_hz: float = 5.0
    automatic_support_stable_samples: int = 3
    arm_home_path: Path | None = None
    robot_id: str | None = None
    execute: bool = False
    target_hz: float = 20.0
    # The RGB inference result and TCP depth frame use independent arrival
    # clocks. On the live GB10 the measured healthy p95 can exceed 100 ms
    # while both sources remain fresh; keep this below the separate 250 ms
    # end-to-end safety gate rather than rejecting every such pair.
    # RGB and depth are received by independent low-latency transports. Their
    # receipt timestamps can differ by almost one depth publish period plus
    # scheduler jitter even though both samples are individually fresh. The
    # bridge still enforces the separate 250 ms perception TTL.
    max_pair_skew_s: float = 0.240
    prediction_horizon_s: float = 0.150
    trajectory_model_path: Path | None = None
    intercept_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not 10.0 <= self.target_hz <= 30.0:
            raise ValueError("target_hz must be between 10 and 30 Hz")
        if not 1.0 <= self.automatic_support_hz <= 10.0:
            raise ValueError("automatic_support_hz must be between 1 and 10 Hz")
        if not 1 <= self.automatic_support_stable_samples <= 12:
            raise ValueError("automatic_support_stable_samples must be between 1 and 12")


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
    """Seed IK from fresh measured joints, then commanded/home fallbacks.

    A disarmed bridge reports its zero-weight release command in
    ``commanded_arm_q``.  Treating that as the physical pose can turn the first
    tracking target into a large jump and immediately trip following-error.
    """

    measured, _ = measured_right_arm_for_intercept(arm_state)
    if measured is not None:
        return list(measured)
    commanded = arm_state.get("commanded_arm_q")
    if isinstance(commanded, list) and len(commanded) == 14:
        return [float(value) for value in commanded[-7:]]
    return [float(value) for value in (home_q or (0.0,) * 7)]


def measured_right_arm_for_intercept(
    arm_state: dict[str, Any],
    *,
    maximum_age_s: float = 0.250,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Return only fresh measured joints; never fall back to a commanded pose."""

    visualization = arm_state.get("visualization") or {}
    if not visualization.get("available"):
        return None, "measured_arm_unavailable"
    try:
        age_ms = float(visualization["state_age_ms"])
    except (KeyError, TypeError, ValueError):
        return None, "measured_arm_age_unavailable"
    if not np.isfinite(age_ms) or age_ms < 0.0 or age_ms > maximum_age_s * 1000.0:
        return None, "measured_arm_stale"
    measured = visualization.get("measured_pose_rad")
    if not isinstance(measured, list) or len(measured) != 29:
        return None, "measured_arm_shape"
    right = np.asarray(measured[22:29], dtype=float)
    if right.shape != (7,) or not np.all(np.isfinite(right)):
        return None, "measured_arm_non_finite"
    return tuple(float(value) for value in right), None


def select_start_escape_waypoint(
    measured_q_rad: Sequence[float],
    path: Sequence[Sequence[float]],
    target_index: int,
    *,
    reached_tolerance_rad: float = 0.050,
    tracking_tolerance_rad: float = 0.050,
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
    if (
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
        self.intercept_profile: LiveInterceptConfig | None = None
        self.intercept_controller: LiveInterceptController | None = None
        if config.intercept_config_path is not None:
            self.intercept_profile = load_live_intercept_config(config.intercept_config_path)
            if self.intercept_profile.calibration_id != self.calibration.calibration_id:
                raise ValueError("intercept profile calibration does not match camera calibration")
            if config.execute and not self.intercept_profile.validated_for_execution:
                raise ValueError(
                    "intercept profile is not validated_for_execution; use dry-run first"
                )
            self.intercept_controller = LiveInterceptController(self.intercept_profile)
        self.tabletop_corners_px: tuple[tuple[float, float], ...] | None = None
        self.tabletop_corner_frame_size: tuple[int, int] | None = None
        self.tabletop_size_m: tuple[float, float] | None = None
        self.tabletop_footprint_error: str | None = None
        if config.tabletop_path is not None:
            try:
                value = json.loads(config.tabletop_path.read_text(encoding="utf-8"))
                frame = value["camera_frame"]
                frame_size = (int(frame["width"]), int(frame["height"]))
                if frame_size[0] <= 0 or frame_size[1] <= 0:
                    raise ValueError("tabletop corner RGB profile is invalid")
                corners = value["corners_px"]
                expected_names = ("near_left", "near_right", "far_right", "far_left")
                if tuple(str(item["name"]) for item in corners) != expected_names:
                    raise ValueError("tabletop corners are not in the required order")
                parsed = tuple((float(item["x"]), float(item["y"])) for item in corners)
                if len(parsed) != 4 or not np.all(np.isfinite(parsed)):
                    raise ValueError("tabletop footprint must contain four finite corners")
                self.tabletop_corners_px = parsed
                self.tabletop_corner_frame_size = frame_size
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
        self.last_depth_preview_at = 0.0
        # A support plane is a live safety input, not a calibration constant.
        # Keep the most recent valid extraction so one noisy RANSAC frame does
        # not release an otherwise healthy tracking session. The fallback is
        # bounded to _SUPPORT_PLANE_GRACE_S and therefore still fails closed.
        self.last_support_plane: Any | None = None
        self.last_support_plane_error: str | None = None
        self.last_support_plane_at = 0.0
        self._automatic_support_last_attempt_at = 0.0
        self._automatic_support_candidate: SupportRegion | None = None
        self._automatic_support_stable_samples = 0
        self._automatic_support_diagnostics: dict[str, object] | None = None
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
        self._approach_path: tuple[tuple[float, ...], ...] | None = None
        self._approach_target_index = 1
        self._approach_target_xyz: tuple[float, float, float] | None = None
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
                    diagnostics = getattr(self.transport, "depth_diagnostics", lambda: {})()
                    self._report(
                        status="waiting_for_depth",
                        reason="depth_receive_timeout",
                        **diagnostics,
                    )
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
        # The depth JPEG is diagnostic UI output, not a control input. Encoding
        # 960x540 at the 20 Hz IK rate consumed several CPU cores and starved
        # YOLO preprocessing while the GPU sat underutilized.
        colormap = None
        if now - self.last_depth_preview_at >= 0.25:
            colormap = depth_colormap_jpeg(aligned, frame.depth_scale)
            self.last_depth_preview_at = now
        base_status: dict[str, Any] = {
            "enabled": True,
            "mode": "execute" if self.config.execute else "dry-run",
            **getattr(self.transport, "depth_diagnostics", lambda: {})(),
            "depth_sequence": frame.sequence,
            "depth_age_ms": round(age_ms, 3),
            "pair_skew_ms": round(skew_s * 1000.0, 3),
            "calibration_id": self.calibration.calibration_id,
            "target_sequence": self.target_sequence,
            "target_age_ms": round(max(0.0, (time.monotonic() - rgb_time) * 1000.0), 3),
            "rgb_frame_id": snapshot.get("rgb_frame_id"),
            "timing_clock_domain": "gb10_monotonic_receipt",
            "localization_filter_latency_ms": None,
            "intercept_planning_latency_ms": None,
            "ik_latency_ms": None,
            "target_publish_latency_ms": None,
        }
        inference_started = snapshot.get("inference_started_monotonic_s")
        inference_completed = snapshot.get("inference_completed_monotonic_s")
        if isinstance(inference_started, (int, float)) and isinstance(
            inference_completed, (int, float)
        ):
            base_status["inference_queue_age_ms"] = round(
                max(0.0, (float(inference_started) - rgb_time) * 1000.0), 3
            )
            base_status["inference_latency_ms"] = round(
                max(0.0, (float(inference_completed) - float(inference_started)) * 1000.0),
                3,
            )
            base_status["inference_result_age_ms"] = round(
                max(0.0, (time.monotonic() - float(inference_completed)) * 1000.0),
                3,
            )
            base_status["inference_to_depth_process_ms"] = round(
                max(0.0, (now - float(inference_completed)) * 1000.0),
                3,
            )
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

        plane: SupportRegion | None = None
        if self.intercept_controller is not None:
            # Interception may continue a latched target through a short
            # detector occlusion, so it needs the current support constraint
            # before track selection. Continuous tracking keeps its original
            # target-loss ordering when interception is disabled.
            plane = self._support_plane(
                aligned,
                frame.depth_scale,
                freeze=arm_state.get("state") in ("ARMING", "ARMED"),
            )
            if plane is not None:
                base_status["visualization"]["support_plane"] = plane.to_dict()
            base_status["visualization"]["support_plane_status"] = {
                "available": plane is not None,
                "age_ms": round(
                    max(0.0, time.monotonic() - self.last_support_plane_at) * 1000.0,
                    1,
                )
                if self.last_support_plane_at
                else None,
                "error": self.last_support_plane_error,
                "source": None if plane is None else plane.source,
                "automatic": self._automatic_support_diagnostics,
                "stable_samples": self._automatic_support_stable_samples,
            }
            if plane is None:
                self._reject(base_status, "support_plane_unavailable", colormap)
                return

        tracks = snapshot.get("tracks") or []
        eligible = [
            item
            for item in tracks
            if item.get("bbox_xyxy")
            and item.get("track_id") is not None
            and float(item.get("confidence", 0.0)) >= 0.25
        ]
        if self.intercept_profile is not None:
            eligible = [
                item
                for item in eligible
                if self.intercept_profile.track_matches(
                    str(item.get("class_name", "")),
                    float(item.get("confidence", 0.0)),
                )
            ]
        selected = next(
            (item for item in eligible if int(item["track_id"]) == self.last_target_track),
            None,
        )
        if selected is None and eligible:
            selected = max(eligible, key=lambda item: float(item.get("confidence", 0.0)))
        if selected is None:
            if self.intercept_controller is not None:
                decision = self.intercept_controller.current_without_observation(now_s=now)
                base_status["intercept"] = self._intercept_status(decision)
                if decision.may_publish and decision.target_palm_position_m is not None:
                    assert plane is not None
                    self._process_committed_intercept_target(
                        np.asarray(decision.target_palm_position_m, dtype=float),
                        decision,
                        base_status,
                        colormap,
                        plane,
                        arm_state,
                    )
                    return
            self.filter.reset()
            self._reject(base_status, "target_lost", colormap)
            return
        if (
            self.intercept_controller is not None
            and self.intercept_controller.active_track_id is not None
            and int(selected["track_id"]) != self.intercept_controller.active_track_id
        ):
            self.filter.reset()
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
        if plane is None:
            plane = self._support_plane(
                aligned,
                frame.depth_scale,
                freeze=arm_state.get("state") in ("ARMING", "ARMED"),
            )
            if plane is not None:
                base_status["visualization"]["support_plane"] = plane.to_dict()
            base_status["visualization"]["support_plane_status"] = {
                "available": plane is not None,
                "age_ms": round(
                    max(0.0, time.monotonic() - self.last_support_plane_at) * 1000.0,
                    1,
                )
                if self.last_support_plane_at
                else None,
                "error": self.last_support_plane_error,
                "source": None if plane is None else plane.source,
                "automatic": self._automatic_support_diagnostics,
                "stable_samples": self._automatic_support_stable_samples,
            }
            if plane is None:
                self._reject(base_status, "support_plane_unavailable", colormap)
                return
        localization_started = time.monotonic()
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
        base_status["localization_filter_latency_ms"] = round(
            max(0.0, (time.monotonic() - localization_started) * 1000.0),
            3,
        )
        base_status.update(
            {
                "track_id": int(selected["track_id"]),
                "object_xyz_m": torso.round(5).tolist(),
                "object_velocity_m_s": tracked.velocity_mps.round(5).tolist(),
                "estimator_consecutive_observations": self.filter.consecutive_observations,
                "estimator_residual_m": round(self.filter.last_residual_m, 6),
                "predicted_xyz_m": predicted.round(5).tolist(),
                "prediction_source": (
                    "learned_trajectory"
                    if learned_prediction is not None
                    else "alpha_beta_fallback"
                ),
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
            }
        )
        intercept_decision: InterceptDecision | None = None
        intercept_measured_q: tuple[float, ...] | None = None
        intercept_publish_allowed = True
        if self.intercept_controller is not None and self.intercept_profile is not None:
            if self.ik is None:
                base_status.update({"ik_status": "unavailable", "ik_error": self.ik_error})
                self._reject(base_status, "ik_unavailable", colormap)
                return
            measured_q, measured_error = measured_right_arm_for_intercept(arm_state)
            if measured_q is None:
                self._reject(base_status, measured_error or "measured_arm_unavailable", colormap)
                return
            intercept_measured_q = measured_q
            if self.intercept_controller.latched_target is None:
                ready_error = float(
                    np.max(
                        np.abs(
                            np.asarray(measured_q)
                            - np.asarray(self.intercept_profile.ready_right_arm_q_rad)
                        )
                    )
                )
                base_status["intercept_ready_pose_error_rad"] = round(ready_error, 6)
            palm_position = self.ik.forward_kinematics(measured_q)[:3, 3]
            intercept_planning_started = time.monotonic()
            intercept_decision = self.intercept_controller.update(
                InterceptObservation(
                    track_id=int(selected["track_id"]),
                    class_name=str(selected.get("class_name", "")),
                    confidence=float(selected.get("confidence", 0.0)),
                    position_m=tuple(float(value) for value in tracked.position_m),
                    velocity_m_s=tuple(float(value) for value in tracked.velocity_mps),
                    timestamp_s=tracked.timestamp_s,
                    consecutive_observations=self.filter.consecutive_observations,
                    residual_m=self.filter.last_residual_m,
                ),
                now_s=now,
                palm_position_m=palm_position,
            )
            base_status["intercept_planning_latency_ms"] = round(
                max(0.0, (time.monotonic() - intercept_planning_started) * 1000.0),
                3,
            )
            base_status["intercept"] = self._intercept_status(intercept_decision)
            if intercept_decision.state in (InterceptState.HOLD, InterceptState.EXPIRED):
                self._reject(
                    base_status,
                    f"intercept_{intercept_decision.reason}",
                    colormap,
                )
                return
            if intercept_decision.target_palm_position_m is None:
                base_status.update(
                    {
                        "status": "intercept_acquiring",
                        "reason": intercept_decision.reason,
                    }
                )
                self.update_status(base_status, colormap)
                return
            target = TargetPose(
                np.asarray(intercept_decision.target_palm_position_m, dtype=float),
                np.asarray((0.0, 0.0, 0.0, 1.0), dtype=float),
            )
            intercept_publish_allowed = intercept_decision.may_publish
        else:
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
                "object_velocity_m_s": tracked.velocity_mps.round(5).tolist(),
                "estimator_consecutive_observations": self.filter.consecutive_observations,
                "estimator_residual_m": round(self.filter.last_residual_m, 6),
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
        last_q = (
            list(intercept_measured_q)
            if intercept_measured_q is not None
            else right_arm_ik_seed(arm_state, self.home_q)
        )
        collision_labels = self.ik.collision_labels(last_q)
        ik_started = time.monotonic()
        if not self.calibration.workspace.contains(target.position):
            self._reject(base_status, "workspace_violation", colormap)
            return
        if not has_support_clearance(target.position, plane, minimum_clearance_m=0.05):
            self._reject(base_status, "support_plane_clearance", colormap)
            return
        ik_step_type = (
            "intercept_local_translation"
            if intercept_decision is not None
            else "analytic_local_translation"
        )
        current_transform = self.ik.forward_kinematics(last_q)
        start_topology = plane.classify_point(
            current_transform[:3, 3],
            side_margin_m=0.10,
            top_clearance_m=0.10,
        )
        base_status["ik_start_topology"] = start_topology
        approach_needed = (
            self._approach_path is not None
            or bool(collision_labels)
            or start_topology != "above_clearance"
        )
        if approach_needed:
            ik_step_type = "adaptive_table_approach"
            if self._approach_path is None:
                transform = current_transform.copy()
                transform[:3, 3] = target.position
                approach = self.ik.plan_adaptive_table_approach(
                    transform,
                    last_q,
                    support_plane=plane,
                )
                if not approach.ok or approach.q_path is None:
                    # The new topology route is deliberately conservative.
                    # Retain the previously validated lift/retract route as a
                    # demo-safe fallback when the observed table footprint is
                    # incomplete or its link envelope rejects an adaptive
                    # edge. It still performs full collision/support checks.
                    approach = self.ik.plan_guided_clearance(
                        last_q,
                        support_plane=plane,
                        lift_m=0.18,
                        forward_m=0.04,
                    )
                    ik_step_type = "guided_table_clearance_fallback"
                    if not approach.ok or approach.q_path is None:
                        base_status.update(
                            {
                                "ik_status": approach.reason,
                                "ik_step_type": ik_step_type,
                                "ik_collision_labels": list(collision_labels),
                                "ik_global_backend": self.ik.global_backend,
                                "ik_local_backend": self.ik.local_backend,
                                "arm_state": arm_state.get("state", "dry-run"),
                            }
                        )
                        self._reject(
                            base_status,
                            f"ik_{approach.reason or 'guided_approach_failed'}",
                            colormap,
                        )
                        return
                self._approach_path = approach.q_path
                self._approach_target_index = 1
                self._approach_target_xyz = tuple(float(value) for value in target.position)
            waypoint, waypoint_index, selection_error = select_start_escape_waypoint(
                last_q,
                self._approach_path,
                self._approach_target_index,
            )
            self._approach_target_index = waypoint_index
            if selection_error is not None:
                self._approach_path = None
                self._approach_target_xyz = None
                self._reject(base_status, f"ik_{selection_error}", colormap)
                return
            if waypoint is not None:
                measured_edge_error = self.ik.validate_joint_path(
                    (last_q, waypoint),
                    support_plane=plane,
                    edge_step_rad=0.010,
                    semantic_edge_step_rad=0.0025,
                    require_escape_cleared=False,
                )
                if measured_edge_error is not None:
                    self._approach_path = None
                    self._approach_target_xyz = None
                    self._reject(
                        base_status,
                        f"ik_measured_edge:{measured_edge_error}",
                        colormap,
                    )
                    return
            if waypoint is None:
                self._approach_path = None
                self._approach_target_index = 1
                self._approach_target_xyz = None
                collision_labels = self.ik.collision_labels(last_q)
                if collision_labels:
                    self._reject(
                        base_status,
                        "ik_approach_complete_but_collision_remains",
                        colormap,
                    )
                    return
                ik_step_type = (
                    "intercept_local_translation"
                    if intercept_decision is not None
                    else "analytic_local_translation"
                )
                transform = self.ik.forward_kinematics(last_q)
                transform[:3, 3] = target.position
                ik = self.ik.solve_local_translation(
                    transform,
                    last_q,
                    support_plane=plane,
                )
            else:
                ik = IKResult(True, waypoint, 0.0, 0.0)
        else:
            # Realtime tracking is an incremental Cartesian servo. Preserve
            # measured wrist orientation and expose one fully validated edge.
            transform = current_transform
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
                "next_edge_collision_checked": bool(ik.ok),
                "ik_collision_labels": list(collision_labels),
                "ik_escape_waypoint": (
                    self._approach_target_index
                    if ik_step_type == "adaptive_table_approach"
                    else None
                ),
                "ik_escape_waypoint_count": (
                    len(self._approach_path)
                    if self._approach_path is not None
                    else None
                ),
                "ik_approach_target_xyz_m": self._approach_target_xyz,
                "ik_global_backend": self.ik.global_backend,
                "ik_local_backend": self.ik.local_backend,
                "ik_position_error_m": ik.position_error_m,
                "ik_orientation_error_rad": ik.orientation_error_rad,
                "arm_state": arm_state.get("state", "dry-run"),
                "arm_weight": arm_state.get("weight"),
                "arm_last_accepted_sequence": arm_state.get("last_sequence"),
                "arm_last_target_age_ms": arm_state.get("last_target_age_ms"),
                "arm_hold_reason": arm_state.get("hold_reason"),
                "arm_fault_reason": arm_state.get("fault_reason"),
                "arm_fault_details": arm_state.get("fault_details"),
                "arm_loop": arm_state.get("loop") or {},
                "ik_latency_ms": round(
                    max(0.0, (time.monotonic() - ik_started) * 1000.0),
                    3,
                ),
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
        if self.config.execute and intercept_publish_allowed:
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
                publish_started = time.monotonic()
                self.transport.publish_target(
                    session_id=str(arm_state["session_id"]),
                    sequence=self.target_sequence,
                    calibration_id=self.calibration.calibration_id,
                    right_arm_q=[float(value) for value in ik.q_rad],
                    pipeline_age_ms=pipeline_age_ms,
                )
                base_status["target_publish_latency_ms"] = round(
                    max(0.0, (time.monotonic() - publish_started) * 1000.0),
                    3,
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
        elif self.config.execute and intercept_decision is not None:
            base_status["status"] = "intercept_preview"
        self.last_target_track = int(selected["track_id"])
        self.update_status(base_status, colormap)

    @staticmethod
    def _intercept_status(decision: InterceptDecision) -> dict[str, Any]:
        return {
            "enabled": True,
            "state": decision.state.value,
            "reason": decision.reason,
            "may_publish": decision.may_publish,
            "target_palm_position_m": (
                None
                if decision.target_palm_position_m is None
                else [round(value, 6) for value in decision.target_palm_position_m]
            ),
            "predicted_crossing_m": (
                None
                if decision.predicted_crossing_m is None
                else [round(value, 6) for value in decision.predicted_crossing_m]
            ),
            "crossing_time_from_now_s": decision.crossing_time_from_now_s,
            "arrival_slack_s": decision.arrival_slack_s,
            "last_confirmation_age_ms": (
                None
                if decision.last_confirmation_age_s is None
                else round(decision.last_confirmation_age_s * 1000.0, 3)
            ),
            "planner": None if decision.plan is None else decision.plan.to_dict(),
        }

    def _process_committed_intercept_target(
        self,
        target_position: np.ndarray,
        decision: InterceptDecision,
        base_status: dict[str, Any],
        colormap: bytes | None,
        plane: SupportRegion,
        arm_state: dict[str, Any],
    ) -> None:
        """Continue one latched Cartesian target through brief bounded occlusion."""

        if self.ik is None:
            base_status.update({"ik_status": "unavailable", "ik_error": self.ik_error})
            self._reject(base_status, "ik_unavailable", colormap)
            return
        measured_q, measured_error = measured_right_arm_for_intercept(arm_state)
        if measured_q is None:
            self._reject(base_status, measured_error or "measured_arm_unavailable", colormap)
            return
        if not self.calibration.workspace.contains(target_position):
            self._reject(base_status, "workspace_violation", colormap)
            return
        if not has_support_clearance(target_position, plane):
            self._reject(base_status, "support_plane_clearance", colormap)
            return
        collision_labels = self.ik.collision_labels(measured_q)
        if collision_labels:
            self._reject(base_status, "intercept_current_pose_collision", colormap)
            return
        ik_started = time.monotonic()
        transform = self.ik.forward_kinematics(measured_q)
        transform[:3, 3] = target_position
        ik = self.ik.solve_local_translation(
            transform,
            measured_q,
            support_plane=plane,
        )
        base_status.update(
            {
                "status": "intercept_occlusion_hold",
                "intercept": self._intercept_status(decision),
                "target_xyz_m": target_position.round(5).tolist(),
                "ik_status": "ok" if ik.ok else ik.reason,
                "ik_step_type": "intercept_local_translation",
                "next_edge_collision_checked": bool(ik.ok),
                "ik_collision_labels": list(collision_labels),
                "ik_position_error_m": ik.position_error_m,
                "ik_orientation_error_rad": ik.orientation_error_rad,
                "arm_state": arm_state.get("state", "dry-run"),
                "arm_weight": arm_state.get("weight"),
                "ik_latency_ms": round(
                    max(0.0, (time.monotonic() - ik_started) * 1000.0),
                    3,
                ),
            }
        )
        if not ik.ok or ik.q_rad is None:
            self._reject(base_status, f"ik_{ik.reason}", colormap)
            return
        base_status["predicted_bounded_arm_command_rad"] = [
            round(float(value), 6) for value in ik.q_rad
        ]
        if self.config.execute:
            if arm_state.get("state") not in ("ARMING", "ARMED") or not arm_state.get(
                "session_id"
            ):
                self._reject(base_status, "arm_not_explicitly_enabled", colormap)
                return
            confirmation_age_s = decision.last_confirmation_age_s
            if confirmation_age_s is None:
                self._reject(base_status, "intercept_confirmation_age_missing", colormap)
                return
            pipeline_age_ms = max(0.0, confirmation_age_s * 1000.0)
            try:
                publish_started = time.monotonic()
                self.transport.publish_target(
                    session_id=str(arm_state["session_id"]),
                    sequence=self.target_sequence,
                    calibration_id=self.calibration.calibration_id,
                    right_arm_q=[float(value) for value in ik.q_rad],
                    pipeline_age_ms=pipeline_age_ms,
                )
                base_status["target_publish_latency_ms"] = round(
                    max(0.0, (time.monotonic() - publish_started) * 1000.0),
                    3,
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
        now = time.monotonic()
        if (
            not freeze
            and self.last_support_plane is not None
            and now - self.last_support_plane_at <= _SUPPORT_PLANE_REFRESH_S
        ):
            return self.last_support_plane
        if (
            freeze
            and self.last_support_plane is not None
            and now - self.last_support_plane_at
            <= _SUPPORT_PLANE_ARMED_TTL_S
        ):
            return self.last_support_plane
        errors: list[str] = []
        if (
            self.config.automatic_support_plane
            and now - self._automatic_support_last_attempt_at
            >= 1.0 / self.config.automatic_support_hz
        ):
            self._automatic_support_last_attempt_at = now
            try:
                candidate, diagnostics = detect_automatic_support_region(
                    aligned,
                    self.calibration.rgb_intrinsics,
                    depth_scale=depth_scale,
                    optical_to_base=self.calibration.optical_to_torso,
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
                    expected_size_m=self.tabletop_size_m,
                )
                self._automatic_support_diagnostics = diagnostics.to_dict()
                if self._support_candidates_agree(
                    self._automatic_support_candidate,
                    candidate,
                ):
                    self._automatic_support_stable_samples += 1
                else:
                    self._automatic_support_candidate = candidate
                    self._automatic_support_stable_samples = 1
                if (
                    self._automatic_support_stable_samples
                    >= self.config.automatic_support_stable_samples
                ):
                    self.last_support_plane = candidate
                    self.last_support_plane_at = now
                    self.last_support_plane_error = None
                    return candidate
                errors.append(
                    "automatic tabletop awaiting temporal consensus "
                    f"{self._automatic_support_stable_samples}/"
                    f"{self.config.automatic_support_stable_samples}"
                )
            except ValueError as exc:
                self._automatic_support_candidate = None
                self._automatic_support_stable_samples = 0
                self._automatic_support_diagnostics = None
                errors.append(f"automatic: {type(exc).__name__}: {exc}")
        try:
            corners_px = self._scaled_tabletop_corners()
            if corners_px is None:
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
                pixel_roi=corners_px,
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
            for u, v in corners_px:
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
            self.last_support_plane_at = now
            self.last_support_plane_error = None
            return support
        except ValueError as exc:
            errors.append(f"clicked fallback: {type(exc).__name__}: {exc}")
            self.last_support_plane_error = "; ".join(errors)
            if (
                self.last_support_plane is not None
                and now - self.last_support_plane_at
                <= _SUPPORT_PLANE_GRACE_S
            ):
                return self.last_support_plane
            if self.config.allow_nominal_support_plane:
                # Explicit free-space test mode: preserve a mathematical
                # exclusion plane even when the physical tabletop has been
                # removed. Its height comes from the validated camera/torso
                # calibration rather than from the current depth image.
                height = self._expected_table_height_m()
                support = SupportRegion.from_xy_bounds(
                    Plane((0.0, 0.0, 1.0), -height),
                    self.calibration.workspace.minimum[:2],
                    self.calibration.workspace.maximum[:2],
                    certified_edges=("u_min", "u_max", "v_min", "v_max"),
                    lateral_margin_m=0.0,
                    source="calibrated_nominal_free_space_plane",
                )
                self.last_support_plane = support
                self.last_support_plane_at = now
                return support
        return None

    def _scaled_tabletop_corners(self) -> tuple[tuple[float, float], ...] | None:
        if self.tabletop_corners_px is None or self.tabletop_corner_frame_size is None:
            return None
        source_width, source_height = self.tabletop_corner_frame_size
        scale_x = self.calibration.rgb_profile.width / source_width
        scale_y = self.calibration.rgb_profile.height / source_height
        return tuple(
            (float(x) * scale_x, float(y) * scale_y)
            for x, y in self.tabletop_corners_px
        )

    @staticmethod
    def _support_candidates_agree(
        previous: SupportRegion | None,
        candidate: SupportRegion,
    ) -> bool:
        if previous is None:
            return False
        normal_agreement = float(np.dot(previous.plane.normal, candidate.plane.normal))
        if normal_agreement < float(np.cos(np.deg2rad(3.0))):
            return False
        if abs(previous.plane.offset - candidate.plane.offset) > 0.02:
            return False
        previous_center = 0.5 * (previous.minimum_uv + previous.maximum_uv)
        candidate_center = 0.5 * (candidate.minimum_uv + candidate.maximum_uv)
        return bool(np.linalg.norm(previous_center - candidate_center) <= 0.05)

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
        cached_support = self.last_support_plane
        cached_support_age_ms = (
            max(0.0, time.monotonic() - self.last_support_plane_at) * 1000.0
            if cached_support is not None and self.last_support_plane_at
            else None
        )
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
            "support_plane": (
                None if cached_support is None else cached_support.to_dict()
            ),
            "support_plane_status": {
                "available": cached_support is not None,
                "age_ms": (
                    None
                    if cached_support_age_ms is None
                    else round(cached_support_age_ms, 1)
                ),
                "error": self.last_support_plane_error,
                "source": (
                    None if cached_support is None else cached_support.source
                ),
                "cached": cached_support is not None,
                "automatic": self._automatic_support_diagnostics,
                "stable_samples": self._automatic_support_stable_samples,
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
        self._start_escape_path = None
        self._start_escape_target_index = 1
        self._approach_path = None
        self._approach_target_index = 1
        self._approach_target_xyz = None

    def reset_intercept(self) -> None:
        if self.intercept_controller is not None:
            self._stop_arm("intercept_reset")
        if self.intercept_controller is not None:
            self.intercept_controller.reset()
        self.filter.reset()
        self.last_target_track = None

    def _reject(self, status: dict[str, Any], reason: str, colormap: bytes | None) -> None:
        if self.intercept_controller is not None and reason != "arm_not_explicitly_enabled":
            if self.intercept_controller.state is InterceptState.COMMITTED:
                decision = self.intercept_controller.invalidate(reason, now_s=time.monotonic())
                status["intercept"] = self._intercept_status(decision)
            elif self.intercept_controller.state not in (
                InterceptState.HOLD,
                InterceptState.EXPIRED,
            ):
                self.intercept_controller.reset()
        status.update({"status": "rejected", "reason": reason})
        if reason not in _SOFT_PERCEPTION_REJECTIONS:
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
