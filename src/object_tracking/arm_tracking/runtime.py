from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable
from urllib import request
from urllib.error import HTTPError, URLError

import numpy as np

from .calibration import Calibration, load_calibration
from .arm_commissioning import load_home_profile
from .depth import DepthFrame, estimate_roi_depth
from .geometry import (
    deproject_depth_samples,
    deproject_pixel,
    extract_support_plane,
    generate_pregrasp_target,
    has_plane_clearance,
)
from .ik_solver import G1RightArmIK, IKUnavailable, default_urdf_path
from .protocol import DepthEnvelopeCodec
from .tracking import PositionVelocityFilter
from .visualization import visualization_state


StatusCallback = Callable[[dict[str, Any], bytes | None], None]
SnapshotCallback = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class RuntimeConfig:
    depth_ws_url: str
    calibration_path: Path
    arm_url: str = "http://192.168.0.213:8766"
    arm_token_file: Path | None = None
    arm_home_path: Path | None = None
    robot_id: str | None = None
    execute: bool = False
    target_hz: float = 15.0
    max_pair_skew_s: float = 0.100
    prediction_horizon_s: float = 0.150
    trajectory_model_path: Path | None = None

    def __post_init__(self) -> None:
        if not 10.0 <= self.target_hz <= 15.0:
            raise ValueError("target_hz must be between 10 and 15 Hz")
        if self.execute and self.arm_token_file is None:
            raise ValueError("--execute requires an arm token file")


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
        repo_root: Path,
    ) -> None:
        self.config = config
        self.snapshot = snapshot
        self.update_status = update_status
        self.repo_root = repo_root
        self.codec = DepthEnvelopeCodec()
        self.calibration = load_calibration(config.calibration_path)
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
        self.last_arm_poll_at = 0.0
        self.last_arm_state: dict[str, Any] = {"state": "unreachable"}
        self.reference_body_q: tuple[float, ...] | None = None
        self.arm_token = self._load_token(config.arm_token_file) if config.execute else None
        self.home_q: tuple[float, ...] | None = None
        if config.arm_home_path is not None:
            profile = load_home_profile(
                config.arm_home_path,
                expected_robot_id=config.robot_id,
            )
            self.home_q = tuple(float(value) for value in profile["measured_q"])
        self.ik: G1RightArmIK | None = None
        self.ik_error: str | None = None
        try:
            self.ik = G1RightArmIK(default_urdf_path(repo_root))
        except IKUnavailable as exc:
            self.ik_error = str(exc)

    @staticmethod
    def _load_token(path: Path | None) -> str:
        if path is None:
            raise ValueError("arm token file is required")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            raise ValueError("arm token file must have mode 0600")
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("arm token file is empty")
        return token

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

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            self._report(status="error", reason=f"websockets unavailable: {exc}")
            return
        retry_s = 0.25
        while not self.stop_event.is_set():
            try:
                with connect(self.config.depth_ws_url, max_size=10 * 1024 * 1024) as websocket:
                    retry_s = 0.25
                    self.last_sequence = -1
                    while not self.stop_event.is_set():
                        envelope = websocket.recv(timeout=1.0)
                        if not isinstance(envelope, bytes):
                            continue
                        receipt = time.monotonic()
                        decoded = self.codec.decode(envelope)
                        if decoded.header.sequence <= self.last_sequence:
                            continue
                        self.last_sequence = decoded.header.sequence
                        self.last_depth = DepthFrame(
                            sequence=decoded.header.sequence,
                            receipt_time_s=receipt,
                            z16=decoded.as_numpy().copy(),
                            depth_scale=decoded.header.depth_scale,
                            calibration_id=decoded.header.calibration_id,
                            sensor_timestamp_ms=decoded.header.sensor_timestamp_ms,
                            registered_to_rgb=decoded.header.registered_to_rgb,
                        )
                        self._process_latest()
            except Exception as exc:
                self._stop_arm("depth_transport_lost")
                self._report(status="waiting_for_depth", reason=f"{type(exc).__name__}: {exc}")
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
        }
        arm_state = self._arm_state()
        base_status["visualization"] = self._visualization_context(arm_state)
        recognized_ids = {
            self.calibration.calibration_id,
            f"factory-{self.calibration.camera_serial}-{frame.z16.shape[1]}x{frame.z16.shape[0]}",
            f"ros-aligned-{frame.z16.shape[1]}x{frame.z16.shape[0]}",
        }
        if (self.config.execute and frame.calibration_id != self.calibration.calibration_id) or (
            not self.config.execute and frame.calibration_id not in recognized_ids
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
        estimate = estimate_roi_depth(aligned, bbox, depth_scale=frame.depth_scale)
        if estimate is None or not estimate.is_certain:
            self._reject(base_status, "depth_uncertain", colormap)
            return
        optical = deproject_pixel(
            estimate.pixel_xy, estimate.depth_m, self.calibration.rgb_intrinsics
        )
        torso = self.calibration.optical_to_torso.apply(optical)
        if self.filter.state is not None and rgb_time <= self.filter.state.timestamp_s:
            self._reject(base_status, "waiting_for_new_rgb_frame", colormap)
            return
        tracked = self.filter.update(torso, rgb_time)
        learned_prediction = None
        if self.learned_forecaster is not None:
            self.learned_forecaster.update(torso, rgb_time)
            learned_prediction = self.learned_forecaster.predict(self.config.prediction_horizon_s)
        predicted = learned_prediction if learned_prediction is not None else tracked.predict(
            self.config.prediction_horizon_s
        )
        target = generate_pregrasp_target(predicted, shoulder_position=(0.0, -0.18, 0.35))
        base_status.update(
            {
                "status": "tracking",
                "track_id": int(selected["track_id"]),
                "depth_valid_fraction": round(estimate.valid_fraction, 4),
                "object_xyz_m": torso.round(5).tolist(),
                "predicted_xyz_m": predicted.round(5).tolist(),
                "prediction_source": (
                    "learned_trajectory" if learned_prediction is not None else "alpha_beta_fallback"
                ),
                "trajectory_model": (
                    None
                    if self.config.trajectory_model_path is None
                    else str(self.config.trajectory_model_path)
                ),
                "trajectory_model_error": self.trajectory_model_error,
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
        if not self.calibration.workspace.contains(target.position):
            self._reject(base_status, "workspace_violation", colormap)
            return
        plane = self._support_plane(aligned, frame.depth_scale)
        if plane is not None:
            base_status["visualization"]["support_plane"] = {
                "normal": plane.normal.tolist(),
                "offset": float(plane.offset),
            }
        if plane is None or not has_plane_clearance(target.position, plane):
            self._reject(base_status, "support_plane_clearance", colormap)
            return
        if self.ik is None:
            base_status.update({"ik_status": "unavailable", "ik_error": self.ik_error})
            self._reject(base_status, "ik_unavailable", colormap)
            return
        last_arm = arm_state.get("commanded_arm_q")
        last_q = (
            last_arm[-7:]
            if isinstance(last_arm, list) and len(last_arm) == 14
            else list(self.home_q or (0.0,) * 7)
        )
        transform = np.eye(4)
        transform[:3, 3] = target.position
        ik = self.ik.solve(transform, last_q, support_plane=plane)
        base_status.update(
            {
                "ik_status": "ok" if ik.ok else ik.reason,
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
            if arm_state.get("state") != "ARMED" or not arm_state.get("session_id"):
                self._reject(base_status, "arm_not_explicitly_enabled", colormap)
                return
            self._post_arm(
                "/arm/target",
                {
                    "session_id": arm_state["session_id"],
                    "sequence": self.target_sequence,
                    "calibration_id": self.calibration.calibration_id,
                    "right_arm_q": list(ik.q_rad),
                    "source_timestamp": time.time(),
                },
            )
            self.target_sequence += 1
            base_status["status"] = "target_sent"
            base_status["target_sequence"] = self.target_sequence
        self.last_target_track = int(selected["track_id"])
        self.update_status(base_status, colormap)

    def _support_plane(self, aligned: np.ndarray, depth_scale: float) -> Any | None:
        try:
            plane, _ = extract_support_plane(
                aligned,
                self.calibration.rgb_intrinsics,
                depth_scale=depth_scale,
                optical_to_base=self.calibration.optical_to_torso,
                stride=8,
            )
            return plane
        except ValueError:
            return None

    def _arm_state(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self.last_arm_poll_at < 0.1:
            return self.last_arm_state
        self.last_arm_poll_at = now
        try:
            with request.urlopen(
                f"{self.config.arm_url.rstrip('/')}/state", timeout=0.2
            ) as response:
                self.last_arm_state = json.loads(response.read())
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
            "measured_object_xyz_m": None,
            "predicted_object_xyz_m": None,
            "predicted_trajectory_xyz_m": [],
            "pregrasp_target_xyz_m": None,
            "end_effector_xyz_m": None,
            "ik_result": None,
        }
        return result

    def _post_arm(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.config.arm_url.rstrip('/')}{path}",
            data=raw,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.arm_token}",
            },
        )
        try:
            with request.urlopen(req, timeout=0.2) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"arm bridge request failed: {exc}") from exc

    def _stop_arm(self, reason: str) -> None:
        if not self.config.execute or self.last_target_track is None:
            return
        try:
            self._post_arm("/arm/stop", {"reason": reason})
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
