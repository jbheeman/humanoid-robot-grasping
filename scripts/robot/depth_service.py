#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from object_tracking.arm_tracking.protocol import DepthEnvelopeCodec
from object_tracking.arm_tracking.calibration import load_calibration


@dataclass(frozen=True)
class CapturedDepth:
    z16: np.ndarray
    sensor_timestamp_ms: float
    timestamp_domain: str


class DepthSource(Protocol):
    calibration: Dict[str, Any]

    def start(self) -> None: ...

    def read(self, timeout_s: float) -> Optional[CapturedDepth]: ...

    def close(self) -> None: ...


class RealSenseDepthSource:
    """Depth-only librealsense source that never claims unverified RGB alignment."""

    def __init__(self, *, width: int, height: int, fps: int, serial: Optional[str]) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline: Any = None
        self.spatial: Any = None
        self.temporal: Any = None
        self.hole_filling: Any = None
        self.calibration: Dict[str, Any] = {}

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Direct depth capture requires pyrealsense2 on the robot") from exc

        context = rs.context()
        devices = context.query_devices()
        if not devices:
            raise RuntimeError("No RealSense device is available")
        device = None
        for candidate in devices:
            candidate_serial = candidate.get_info(rs.camera_info.serial_number)
            if self.serial is None or candidate_serial == self.serial:
                device = candidate
                self.serial = candidate_serial
                break
        if device is None:
            raise RuntimeError(f"RealSense serial {self.serial!r} was not found")

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline = rs.pipeline(context)
        profile = self.pipeline.start(config)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = depth_profile.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        color_profiles: List[Dict[str, Any]] = []
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                try:
                    video = stream_profile.as_video_stream_profile()
                    if video.stream_type() != rs.stream.color:
                        continue
                    extrinsics = depth_profile.get_extrinsics_to(video)
                    color_intrinsics = video.get_intrinsics()
                    color_profiles.append(
                        {
                            "width": color_intrinsics.width,
                            "height": color_intrinsics.height,
                            "fps": video.fps(),
                            "format": str(video.format()),
                            "intrinsics": _intrinsics_dict(color_intrinsics),
                            "T_color_depth": {
                                "rotation_row_major": list(extrinsics.rotation),
                                "translation_m": list(extrinsics.translation),
                            },
                        }
                    )
                except Exception:
                    continue

        self.calibration = {
            "schema_version": 1,
            "source": "librealsense_depth_only",
            "aligned_to_rgb": False,
            "registration_validated": False,
            "camera_serial": self.serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "depth_profile": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fps": self.fps,
                "format": "z16",
                "intrinsics": _intrinsics_dict(intrinsics),
                "depth_scale": depth_scale,
            },
            "factory_color_profiles": color_profiles,
            "calibration_id": f"factory-{self.serial}-{intrinsics.width}x{intrinsics.height}",
        }
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def read(self, timeout_s: float) -> Optional[CapturedDepth]:
        if self.pipeline is None:
            raise RuntimeError("Depth source is not started")
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=max(1, int(timeout_s * 1000)))
        except RuntimeError:
            return None
        depth = frames.get_depth_frame()
        if not depth:
            return None
        for filter_ in (self.spatial, self.temporal, self.hole_filling):
            depth = filter_.process(depth)
        z16 = np.asanyarray(depth.get_data()).astype("<u2", copy=True)
        return CapturedDepth(
            z16=z16,
            sensor_timestamp_ms=float(depth.get_timestamp()),
            timestamp_domain=str(depth.get_frame_timestamp_domain()),
        )

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class RosAlignedDepthSource:
    """ROS2 aligned-depth subscriber with latest-frame-only buffering."""

    def __init__(
        self,
        *,
        image_topic: str,
        camera_info_topic: str,
        depth_scale: float,
        startup_timeout_s: float = 3.0,
    ) -> None:
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.depth_scale = depth_scale
        self.startup_timeout_s = startup_timeout_s
        self.calibration: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._latest: Optional[CapturedDepth] = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: Optional[threading.Thread] = None
        self._rclpy: Any = None
        self._camera_info: Any = None

    def start(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS aligned depth is unavailable (rclpy/sensor_msgs missing)"
            ) from exc
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("g1_aligned_depth_bridge")
        self._node.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self._node.create_subscription(
            Image,
            self.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True, name="ros-depth")
        self._thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None and self._camera_info is not None:
                    return
            time.sleep(0.05)
        self.close()
        raise RuntimeError(f"No aligned Z16 frames arrived on {self.image_topic}")

    def _on_camera_info(self, message: Any) -> None:
        with self._lock:
            self._camera_info = message
            self._refresh_calibration()

    def _on_image(self, message: Any) -> None:
        if str(message.encoding).upper() not in {"16UC1", "MONO16"}:
            return
        expected = int(message.width) * int(message.height) * 2
        raw = bytes(message.data)
        if len(raw) < expected:
            return
        dtype = ">u2" if int(message.is_bigendian) else "<u2"
        z16 = np.frombuffer(raw[:expected], dtype=dtype).astype("<u2", copy=True)
        z16 = z16.reshape(int(message.height), int(message.width))
        stamp = message.header.stamp
        timestamp_ms = float(stamp.sec) * 1000.0 + float(stamp.nanosec) / 1_000_000.0
        with self._lock:
            self._latest = CapturedDepth(z16, timestamp_ms, "ros_time")
            self._refresh_calibration(width=int(message.width), height=int(message.height))

    def _refresh_calibration(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        info = self._camera_info
        if info is None:
            return
        width = width or int(info.width)
        height = height or int(info.height)
        self.calibration = {
            "schema_version": 1,
            "source": "ros_aligned_depth",
            "aligned_to_rgb": True,
            "registration_validated": False,
            "camera_serial": None,
            "depth_profile": {
                "width": width,
                "height": height,
                "format": "z16",
                "intrinsics": {
                    "width": int(info.width),
                    "height": int(info.height),
                    "fx": float(info.k[0]),
                    "fy": float(info.k[4]),
                    "ppx": float(info.k[2]),
                    "ppy": float(info.k[5]),
                    "distortion_model": str(info.distortion_model),
                    "distortion_coefficients": [float(value) for value in info.d],
                },
                "depth_scale": self.depth_scale,
            },
            "image_topic": self.image_topic,
            "camera_info_topic": self.camera_info_topic,
            "calibration_id": f"ros-aligned-{width}x{height}",
        }

    def read(self, timeout_s: float) -> Optional[CapturedDepth]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        self._executor = None
        self._thread = None
        self._node = None


class AutoDepthSource:
    def __init__(self, preferred: DepthSource, fallback: DepthSource) -> None:
        self.preferred = preferred
        self.fallback = fallback
        self.active: Optional[DepthSource] = None
        self.calibration: Dict[str, Any] = {}
        self.preferred_error: Optional[str] = None

    def start(self) -> None:
        try:
            self.preferred.start()
            self.active = self.preferred
        except Exception as exc:
            self.preferred_error = f"{type(exc).__name__}: {exc}"
            self.fallback.start()
            self.active = self.fallback
        self.calibration = dict(self.active.calibration)
        self.calibration["preferred_source_error"] = self.preferred_error

    def read(self, timeout_s: float) -> Optional[CapturedDepth]:
        if self.active is None:
            raise RuntimeError("Depth source is not started")
        return self.active.read(timeout_s)

    def close(self) -> None:
        if self.active is not None:
            self.active.close()
            self.active = None


def _intrinsics_dict(intrinsics: Any) -> Dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
    }


class DepthService:
    def __init__(self, source: DepthSource, *, transmit_fps: float = 15.0) -> None:
        self.source = source
        self.transmit_fps = transmit_fps
        self.codec = DepthEnvelopeCodec()
        self.lock = threading.Lock()
        self.latest_envelope: Optional[bytes] = None
        self.latest_sequence = -1
        self.latest_received_monotonic: Optional[float] = None
        self.last_error: Optional[str] = None
        self.started_at = time.monotonic()
        self.frames_captured = 0
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self, calibration_path: Optional[str] = None) -> None:
        self.source.start()
        if calibration_path:
            bind_validated_calibration(self.source, calibration_path)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="realsense-depth"
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.source.close()

    def _capture_loop(self) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            try:
                frame = self.source.read(timeout_s=0.5)
                if frame is None:
                    continue
                calibration = self.source.calibration
                profile = calibration["depth_profile"]
                envelope = self.codec.encode(
                    frame.z16,
                    sequence=sequence,
                    width=int(profile["width"]),
                    height=int(profile["height"]),
                    depth_scale=float(profile["depth_scale"]),
                    sensor_timestamp_ms=frame.sensor_timestamp_ms,
                    timestamp_domain=frame.timestamp_domain,
                    calibration_id=str(calibration["calibration_id"]),
                    registered_to_rgb=bool(calibration.get("aligned_to_rgb", False)),
                )
                with self.lock:
                    self.latest_envelope = envelope
                    self.latest_sequence = sequence
                    self.latest_received_monotonic = time.monotonic()
                    self.frames_captured += 1
                    self.last_error = None
                sequence += 1
            except Exception as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.1)

    def health(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            age = (
                None
                if self.latest_received_monotonic is None
                else now - self.latest_received_monotonic
            )
            fresh = age is not None and age <= max(0.25, 3.0 / self.transmit_fps)
            return {
                "ok": fresh and self.last_error is None,
                "service": "unitree_realsense_depth",
                "sensor_fresh": fresh,
                "executable": fresh
                and bool(self.source.calibration.get("registration_validated", False)),
                "sequence": self.latest_sequence,
                "frame_age_ms": None if age is None else round(age * 1000.0, 3),
                "frames_captured": self.frames_captured,
                "transmit_fps": self.transmit_fps,
                "last_error": self.last_error,
                "uptime_s": round(now - self.started_at, 3),
                "calibration_id": self.source.calibration.get("calibration_id"),
                "aligned_to_rgb": self.source.calibration.get("aligned_to_rgb", False),
                "registration_validated": self.source.calibration.get(
                    "registration_validated", False
                ),
            }


def create_app(service: DepthService) -> FastAPI:
    app = FastAPI(title="Unitree G1 RealSense Depth Service", version="1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return service.health()

    @app.get("/depth/calibration")
    def calibration() -> Dict[str, Any]:
        return service.source.calibration

    @app.websocket("/depth/stream")
    async def depth_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        last_sequence = -1
        period = 1.0 / service.transmit_fps
        try:
            while True:
                with service.lock:
                    sequence = service.latest_sequence
                    envelope = service.latest_envelope
                if envelope is not None and sequence != last_sequence:
                    await websocket.send_bytes(envelope)
                    last_sequence = sequence
                await _async_sleep(period)
        except WebSocketDisconnect:
            return

    return app


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot-local RealSense Z16 depth service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--transmit-fps", type=float, default=15.0)
    parser.add_argument(
        "--source",
        choices=("auto", "ros", "librealsense"),
        default="auto",
        help="Prefer ROS aligned depth and fall back to direct depth-only capture in auto mode.",
    )
    parser.add_argument(
        "--ros-image-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--ros-camera-info-topic",
        default="/camera/camera/aligned_depth_to_color/camera_info",
    )
    parser.add_argument("--ros-depth-scale", type=float, default=0.001)
    parser.add_argument(
        "--calibration",
        help="Validated calibration YAML used to bind streamed frames to an execution ID",
    )
    return parser


def bind_validated_calibration(source: DepthSource, calibration_path: str) -> None:
    calibration = load_calibration(calibration_path)
    active = source.calibration
    profile = active.get("depth_profile", {})
    active_shape = (int(profile.get("width", 0)), int(profile.get("height", 0)))
    expected_shape = (
        (calibration.rgb_profile.width, calibration.rgb_profile.height)
        if active.get("aligned_to_rgb")
        else (calibration.depth_profile.width, calibration.depth_profile.height)
    )
    if active_shape != expected_shape:
        raise RuntimeError(
            f"Active depth profile {active_shape} does not match calibration {expected_shape}"
        )
    active_serial = active.get("camera_serial")
    if active_serial is not None and active_serial != calibration.camera_serial:
        raise RuntimeError("Active RealSense serial does not match calibration")
    if not math.isclose(
        float(profile.get("depth_scale", 0.0)), calibration.depth_scale, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("Active depth scale does not match calibration")
    calibration.validate_for_execution(
        camera_serial=str(active_serial or calibration.camera_serial),
        camera_firmware=active.get("firmware"),
        rgb_profile=calibration.rgb_profile,
        depth_profile=calibration.depth_profile,
        waist_rad=calibration.waist_reference_rad,
        calibration_id=calibration.calibration_id,
    )
    active["calibration_id"] = calibration.calibration_id
    active["registration_validated"] = True
    active["camera_serial"] = calibration.camera_serial
    active["calibration_hash"] = calibration.calibration_hash


def main() -> None:
    args = build_parser().parse_args()
    if args.capture_fps <= 0 or args.transmit_fps <= 0 or not math.isfinite(args.transmit_fps):
        raise SystemExit("FPS values must be finite and positive")
    direct_source = RealSenseDepthSource(
        width=args.width,
        height=args.height,
        fps=args.capture_fps,
        serial=args.serial,
    )
    ros_source = RosAlignedDepthSource(
        image_topic=args.ros_image_topic,
        camera_info_topic=args.ros_camera_info_topic,
        depth_scale=args.ros_depth_scale,
    )
    if args.source == "ros":
        source: DepthSource = ros_source
    elif args.source == "librealsense":
        source = direct_source
    else:
        source = AutoDepthSource(ros_source, direct_source)
    service = DepthService(source, transmit_fps=args.transmit_fps)
    service.start(args.calibration)
    app = create_app(service)
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
