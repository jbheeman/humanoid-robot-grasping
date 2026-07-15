#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from object_tracking.arm_tracking.protocol import DepthEnvelopeCodec
from object_tracking.arm_tracking.calibration import load_calibration


@dataclass(frozen=True)
class CapturedDepth:
    z16: np.ndarray
    sensor_timestamp_ms: float
    timestamp_domain: str
    color_bgr: np.ndarray | None = None


class DepthSource(Protocol):
    calibration: Dict[str, Any]

    def start(self) -> None: ...

    def read(self, timeout_s: float) -> Optional[CapturedDepth]: ...

    def close(self) -> None: ...


class RealSenseDepthSource:
    """One D435I pipeline providing colour and depth aligned to that colour stream."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        serial: Optional[str],
        color_width: int = 960,
        color_height: int = 540,
        color_fps: int = 60,
        enable_color: bool = False,
        registered_to_output_rgb: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.enable_color = enable_color
        self.registered_to_output_rgb = registered_to_output_rgb
        self.pipeline: Any = None
        self.align: Any = None
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
        if self.enable_color:
            config.enable_stream(
                rs.stream.color,
                self.color_width,
                self.color_height,
                rs.format.bgr8,
                self.color_fps,
            )
        self.pipeline = rs.pipeline(context)
        profile = self.pipeline.start(config)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
            if self.enable_color
            else None
        )
        intrinsics = (
            color_profile.get_intrinsics()
            if color_profile is not None
            else depth_profile.get_intrinsics()
        )
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
            "source": (
                "librealsense_aligned_color_depth"
                if self.enable_color
                else "librealsense_depth_only"
            ),
            "aligned_to_rgb": self.registered_to_output_rgb,
            "registration_validated": self.registered_to_output_rgb,
            "camera_serial": self.serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "depth_profile": {
                "width": (
                    color_profile.width() if color_profile is not None else depth_profile.width()
                ),
                "height": (
                    color_profile.height() if color_profile is not None else depth_profile.height()
                ),
                "fps": self.fps,
                "format": "z16",
                "intrinsics": _intrinsics_dict(intrinsics),
                "depth_scale": depth_scale,
            },
            "color_profile": None
            if color_profile is None
            else {
                "width": color_profile.width(),
                "height": color_profile.height(),
                "fps": self.color_fps,
                "format": "bgr8",
                "intrinsics": _intrinsics_dict(intrinsics),
            },
            "factory_color_profiles": color_profiles,
            "calibration_id": (
                f"d435i-aligned-{self.serial}-{color_profile.width()}x{color_profile.height()}"
                if color_profile is not None
                else f"factory-{self.serial}-{depth_profile.width()}x{depth_profile.height()}"
            ),
        }
        self.align = rs.align(rs.stream.color) if self.enable_color else None
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
        aligned = self.align.process(frames) if self.align is not None else frames
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame() if self.enable_color else None
        if not depth or (self.enable_color and not color):
            return None
        for filter_ in (self.spatial, self.temporal, self.hole_filling):
            depth = filter_.process(depth)
        z16 = np.asanyarray(depth.get_data()).astype("<u2", copy=True)
        return CapturedDepth(
            z16=z16,
            sensor_timestamp_ms=float(depth.get_timestamp()),
            timestamp_domain=str(depth.get_frame_timestamp_domain()),
            color_bgr=None if color is None else np.asanyarray(color.get_data()).copy(),
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

    def _refresh_calibration(
        self, width: Optional[int] = None, height: Optional[int] = None
    ) -> None:
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
    def __init__(
        self,
        source: DepthSource,
        *,
        transmit_fps: float = 15.0,
        rgb_relay: "RealSenseRgbRtpRelay | None" = None,
    ) -> None:
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
        self.source_restarts = 0
        self.rgb_relay = rgb_relay
        self._calibration_path: Optional[str] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self, calibration_path: Optional[str] = None) -> None:
        self._calibration_path = calibration_path
        self.source.start()
        try:
            if calibration_path:
                bind_validated_calibration(self.source, calibration_path)
            if self.rgb_relay is not None:
                color = self.source.calibration.get("color_profile")
                if not isinstance(color, dict):
                    raise RuntimeError("RGB relay requires a source with a color profile")
                self.rgb_relay.start(width=int(color["width"]), height=int(color["height"]))
        except Exception:
            if self.rgb_relay is not None:
                self.rgb_relay.close()
            self.source.close()
            raise
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
        if self.rgb_relay is not None:
            self.rgb_relay.close()

    def _capture_loop(self) -> None:
        sequence = 0
        consecutive_timeouts = 0
        # Capturing at the camera rate is useful for freshness, but serialising
        # every 60 Hz Z16 image when the ROS transport is configured for 15 Hz
        # starves the ROS executor on the robot.  Keep the newest camera frame
        # and only encode frames at the requested transport rate.
        next_transmit_at = 0.0
        while not self.stop_event.is_set():
            try:
                frame = self.source.read(timeout_s=0.5)
                if frame is None:
                    consecutive_timeouts += 1
                    # librealsense can leave a pipeline open but stop yielding
                    # frames after a USB/camera hiccup. Recreate it instead of
                    # publishing one stale frame forever.
                    if consecutive_timeouts >= 4:
                        self._restart_source()
                        consecutive_timeouts = 0
                    continue
                consecutive_timeouts = 0
                now = time.monotonic()
                if now < next_transmit_at:
                    continue
                next_transmit_at = now + (1.0 / self.transmit_fps)
                if self.rgb_relay is not None and frame.color_bgr is not None:
                    self.rgb_relay.write(frame.color_bgr)
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

    def _restart_source(self) -> None:
        try:
            self.source.close()
            if self.stop_event.is_set():
                return
            self.source.start()
            if self._calibration_path:
                bind_validated_calibration(self.source, self._calibration_path)
            with self.lock:
                self.source_restarts += 1
                self.last_error = "depth source stalled; restarting capture"
        except Exception as exc:
            with self.lock:
                self.last_error = f"depth source restart failed: {type(exc).__name__}: {exc}"
            self.stop_event.wait(0.5)

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
                "source_restarts": self.source_restarts,
                "transmit_fps": self.transmit_fps,
                "last_error": self.last_error,
                "uptime_s": round(now - self.started_at, 3),
                "calibration_id": self.source.calibration.get("calibration_id"),
                "aligned_to_rgb": self.source.calibration.get("aligned_to_rgb", False),
                "registration_validated": self.source.calibration.get(
                    "registration_validated", False
                ),
                "rgb_relay": None if self.rgb_relay is None else self.rgb_relay.health(),
            }


class RealSenseRgbRtpRelay:
    """Low-latency H264 RTP output fed by the same D435I frames as depth."""

    def __init__(self, *, host: str, port: int, fps: int, bitrate: int = 8_000_000) -> None:
        if not host:
            raise ValueError("RGB relay host is required")
        if not (0 < port < 65536 and fps > 0 and bitrate > 0):
            raise ValueError("invalid RGB relay configuration")
        self.host = host
        self.port = port
        self.fps = fps
        self.bitrate = bitrate
        self.process: subprocess.Popen[bytes] | None = None
        self.frames_sent = 0
        self.last_error: str | None = None

    def start(self, *, width: int, height: int) -> None:
        command = [
            "gst-launch-1.0",
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            "videoparse",
            "format=bgr",
            f"width={width}",
            f"height={height}",
            f"framerate={self.fps}/1",
            "!",
            "queue",
            "max-size-buffers=1",
            "max-size-time=0",
            "max-size-bytes=0",
            "leaky=downstream",
            "!",
            "videoconvert",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            f"bitrate={self.bitrate}",
            f"iframeinterval={self.fps}",
            f"idrinterval={self.fps}",
            "insert-sps-pps=1",
            "!",
            "h264parse",
            "!",
            "rtph264pay",
            "pt=96",
            "config-interval=1",
            "!",
            "udpsink",
            f"host={self.host}",
            f"port={self.port}",
            "sync=false",
            "async=false",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                bufsize=0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.process = None
            raise RuntimeError(f"Could not start the D435I H264 RTP relay: {exc}") from exc

    def write(self, frame: np.ndarray) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is not None or self.process.stdin is None:
                raise RuntimeError(f"GStreamer relay exited with code {self.process.returncode}")
            contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
            self.process.stdin.write(contiguous.tobytes())
            self.frames_sent += 1
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def health(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "frames_sent": self.frames_sent,
            "last_error": self.last_error,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robot-local RealSense Z16 source used by the ROS 2 robot node"
    )
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
    raise SystemExit(
        "The standalone depth HTTP server was removed. Start the ROS 2 robot node with "
        "`uv run g1 robot start`; it publishes compressed depth on /g1/depth."
    )


if __name__ == "__main__":
    main()
