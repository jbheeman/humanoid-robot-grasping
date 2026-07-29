from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
import re
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from object_tracking.research_session import ResearchSession
from object_tracking.simple_tracker import SimpleTracker


def unitree_g1_videohub_pipeline(
    device: str, width: int = 1280, height: int = 720, out_width: int = 640, out_height: int = 360
) -> str:
    device_path = device if device.startswith("/") else f"/dev/{device}"
    return (
        f"v4l2src device={device_path} io-mode=2 do-timestamp=true ! "
        f"image/jpeg,width={width},height={height},framerate=30/1 ! "
        "jpegdec ! videoconvert ! videoscale ! "
        f"video/x-raw,width={out_width},height={out_height},format=BGR ! "
        "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
        "appsink sync=false drop=true max-buffers=1"
    )


UNITREE_UDP_PIPELINE = (
    "udpsrc port=5600 buffer-size=1048576 ! "
    "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! "
    "rtpjitterbuffer latency=20 drop-on-latency=true ! "
    "rtph264depay ! h264parse ! avdec_h264 max-threads=8 ! videoconvert ! videoscale ! "
    "video/x-raw,width=640,height=360,format=BGR ! "
    "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
    "appsink sync=false drop=true max-buffers=1"
)

DEFAULT_CAMERA_NAME = os.environ.get("G1_CAMERA_NAME", "main")
DEFAULT_CAMERA_DEVICE = os.environ.get("G1_CAMERA_DEVICE", "videohub_pc4")
DEFAULT_PIPELINE = os.environ.get("G1_CAMERA_PIPELINE") or unitree_g1_videohub_pipeline(
    DEFAULT_CAMERA_DEVICE
)
REPO_ROOT = Path(__file__).resolve().parents[2]
GB10_WEB_DIR = REPO_ROOT / "scripts" / "gb10" / "web"
COMMISSIONING_TEMPLATE = REPO_ROOT / "scripts" / "robot" / "web" / "arm_commissioning.html"
TABLETOP_CALIBRATION_PATH = REPO_ROOT / "runs" / "localization" / "tabletop-rectangle.json"


@dataclass(frozen=True)
class ProcessedFrameBundle:
    """One immutable detector result tied to the exact copied RGB frame."""

    frame_id: int
    frame_receipt_monotonic_s: float
    frame_shape: tuple[int, ...]
    inference_started_monotonic_s: float
    inference_completed_monotonic_s: float
    detections: tuple[dict[str, Any], ...]
    tracks: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("processed frame_id must be non-negative")
        timing = (
            self.frame_receipt_monotonic_s,
            self.inference_started_monotonic_s,
            self.inference_completed_monotonic_s,
        )
        if (
            not all(np.isfinite(value) and value >= 0.0 for value in timing)
            or self.inference_started_monotonic_s < self.frame_receipt_monotonic_s
            or self.inference_completed_monotonic_s < self.inference_started_monotonic_s
        ):
            raise ValueError("processed frame monotonic timing is invalid")
        if len(self.frame_shape) not in (2, 3) or any(value <= 0 for value in self.frame_shape):
            raise ValueError("processed frame shape is invalid")
        if any(int(track.get("missed_updates", 0)) != 0 for track in self.tracks):
            raise ValueError("processed frame contains a retained stale track")


def current_detection_tracks(
    tracks: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Copy only tracks supported by a detection in this inference batch."""

    return tuple(
        dict(track)
        for track in tracks
        if int(track.get("missed_updates", 0)) == 0
    )


def processed_frame_is_newer(
    existing: ProcessedFrameBundle | None,
    candidate: ProcessedFrameBundle,
) -> bool:
    """Reject duplicate/out-of-order detector publications."""

    return existing is None or candidate.frame_id > existing.frame_id


@dataclass
class SharedState:
    raw_frame: Optional[np.ndarray] = None
    raw_frame_received_monotonic: float = 0.0
    annotated_frame: Optional[np.ndarray] = None
    jpeg_bytes: Optional[bytes] = None
    latest_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_tracks: list[dict[str, Any]] = field(default_factory=list)
    processed_frame: Optional[ProcessedFrameBundle] = None
    tabletop_localization: dict[str, Any] = field(
        default_factory=lambda: {"configured": False, "status": "not_configured"}
    )
    last_error: Optional[str] = None
    frame_count: int = 0
    jpeg_frame_id: int = 0
    yolo_count: int = 0
    encode_count: int = 0
    fps: float = 0.0
    yolo_fps: float = 0.0
    encode_fps: float = 0.0
    stream_fps_limit: float = 0.0
    expected_fps: float = 0.0
    camera_pipeline: str = DEFAULT_PIPELINE
    camera_name: str = DEFAULT_CAMERA_NAME
    capture_backend: str = ""
    model_name: str = "none"
    inference_status: str = "disabled"
    inference_error: Optional[str] = None
    inference_warmup_seconds: float = 0.0
    opencv_threads: int = 0
    torch_threads: int = 0
    capture_session_dir: Optional[Path] = None
    capture_count: int = 0
    arm_tracking: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "mode": "dry-run",
            "status": "disabled",
            "reason": "start with --calibration to enable ROS 2 depth tracking",
        }
    )
    depth_colormap_jpeg: Optional[bytes] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


state = SharedState()
frame_ready = threading.Condition(state.lock)
jpeg_ready = threading.Condition(state.lock)
tracker = SimpleTracker()
_tabletop_cache_lock = threading.Lock()
_tabletop_cache_mtime: float | None = None
_tabletop_cache: dict[str, Any] | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    yield
    shutdown_ros_transport()


app = FastAPI(lifespan=app_lifespan)
app.mount("/visual", StaticFiles(directory=GB10_WEB_DIR / "visual"), name="visual")
research_session: ResearchSession | None = None
tracking_transport: Any | None = None
arm_runtime: Any | None = None
depth_preview_thread: threading.Thread | None = None
depth_preview_stop = threading.Event()


def yolo_enabled(model_name: str) -> bool:
    return model_name.strip().lower() not in {"", "none", "off", "disabled"}


def optional_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def torch_cuda_available() -> bool:
    torch = optional_torch()
    return bool(torch is not None and torch.cuda.is_available())


def require_inference_stack() -> tuple[Any, Any]:
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "YOLO inference requires the train dependency group. Run: uv sync --group vision --group train"
        ) from exc
    return torch, YOLO


def opencv_gstreamer_enabled() -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return hasattr(cv2, "CAP_GSTREAMER")

    for line in info.splitlines():
        if line.strip().startswith("GStreamer:"):
            return "YES" in line.upper()
    return False


def _pipeline_dimension(pipeline: str, name: str, default: int) -> int:
    matches = re.findall(rf"{name}=\(int\)(\d+)|{name}=(\d+)", pipeline)
    if not matches:
        return default
    value = matches[-1][0] or matches[-1][1]
    return int(value)


def _strip_appsink(pipeline: str) -> str:
    return re.sub(r"\s*!\s*appsink(?:\s+[^!]*)?$", "", pipeline).strip()


class GstLaunchCapture:
    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline
        self.width = _pipeline_dimension(pipeline, "width", 640)
        self.height = _pipeline_dimension(pipeline, "height", 360)
        self.frame_size = self.width * self.height * 3
        self.command = f"gst-launch-1.0 -q {_strip_appsink(pipeline)} ! fdsink fd=1"
        self.proc = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def isOpened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.proc.stdout is None:
            return False, None

        chunks = []
        remaining = self.frame_size
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                return False, None
            chunks.append(chunk)
            remaining -= len(chunk)

        frame = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        return True, frame.reshape((self.height, self.width, 3)).copy()

    def release(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def open_capture(pipeline: str, capture_backend: str = "auto") -> tuple[Any, str]:
    if capture_backend == "gst-launch":
        return GstLaunchCapture(pipeline), "gst-launch"

    if capture_backend == "opencv":
        if not opencv_gstreamer_enabled():
            raise RuntimeError(
                "OpenCV GStreamer backend requested, but cv2 was built without GStreamer."
            )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("OpenCV GStreamer backend could not open the camera pipeline.")
        return cap, "opencv-gstreamer"

    if capture_backend != "auto":
        raise ValueError(f"Unknown capture backend: {capture_backend}")

    if opencv_gstreamer_enabled():
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "opencv-gstreamer"
        cap.release()

    print("OpenCV GStreamer is unavailable; falling back to gst-launch-1.0 raw frame pipe.")
    return GstLaunchCapture(pipeline), "gst-launch"


def draw_fps(frame: np.ndarray, fps: float, yolo_fps: float) -> None:
    text = f"camera fps={fps:.1f} | yolo fps={yolo_fps:.1f}"
    cv2.putText(
        frame,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_tracks(frame: np.ndarray, tracks: list[dict[str, Any]]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track["bbox_xyxy"]]
        cx, cy = [int(v) for v in track["center_xy"]]
        vx, vy = track["velocity_px_per_sec"]
        label = f"id={track['track_id']} {track['class_name']} vx={vx:.0f} vy={vy:.0f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (int(cx + vx * 0.25), int(cy + vy * 0.25)),
            (0, 200, 255),
            2,
            tipLength=0.25,
        )
        cv2.putText(
            frame,
            label,
            (x1, min(y2 + 18, frame.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )


def make_detection(
    box: Any,
    names: dict[int, str],
    timestamp: float,
    frame_number: int,
) -> dict[str, Any]:
    xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
    cls_id = int(box.cls[0].item())
    score = float(box.conf[0].item())
    x1, y1, x2, y2 = xyxy

    return {
        "class_id": cls_id,
        "class_name": str(names.get(cls_id, cls_id)),
        "confidence": score,
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "timestamp": timestamp,
        "frame_number": frame_number,
    }


def configure_runtime(opencv_threads: int, torch_threads: int, needs_torch: bool) -> None:
    if opencv_threads > 0:
        cv2.setNumThreads(opencv_threads)

    torch_threads_actual = 0
    if needs_torch:
        torch = optional_torch()
        if torch is not None:
            if torch_threads > 0:
                torch.set_num_threads(torch_threads)
                try:
                    torch.set_num_interop_threads(max(1, min(4, torch_threads)))
                except RuntimeError:
                    pass
            torch_threads_actual = torch.get_num_threads()

    with state.lock:
        state.opencv_threads = cv2.getNumThreads()
        state.torch_threads = torch_threads_actual

    print("OpenCV threads:", cv2.getNumThreads())
    if needs_torch:
        print("Torch threads:", torch_threads_actual if torch_threads_actual else "unavailable")


def camera_loop(pipeline: str, camera_name: str, capture_backend: str) -> None:
    print("OpenCV:", cv2.__version__)
    print("OpenCV GStreamer enabled:", opencv_gstreamer_enabled())
    print("Camera name:", camera_name)
    print("Opening pipeline:")
    print(pipeline)
    print("Requested capture backend:", capture_backend)

    with state.lock:
        state.camera_pipeline = pipeline
        state.camera_name = camera_name
        state.capture_backend = capture_backend

    try:
        cap, backend = open_capture(pipeline, capture_backend)
    except (RuntimeError, ValueError) as exc:
        with frame_ready:
            state.last_error = str(exc)
            frame_ready.notify_all()
        print(f"ERROR: {exc}")
        return

    with state.lock:
        state.capture_backend = backend
    print("Capture backend:", backend)

    if not cap.isOpened():
        with frame_ready:
            state.last_error = f"Could not open GStreamer camera pipeline with {backend}"
            frame_ready.notify_all()
        print(f"ERROR: Could not open GStreamer camera pipeline with {backend}")
        return

    last_fps_time = time.time()
    frames_since_fps = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            with frame_ready:
                state.last_error = "No frame received from camera"
                frame_ready.notify_all()
            time.sleep(0.001)
            continue

        frames_since_fps += 1
        now = time.time()
        cam_fps = None
        if now - last_fps_time >= 1.0:
            cam_fps = frames_since_fps / (now - last_fps_time)
            frames_since_fps = 0
            last_fps_time = now

        with frame_ready:
            state.raw_frame = frame
            state.raw_frame_received_monotonic = time.monotonic()
            state.frame_count += 1
            if cam_fps is not None:
                state.fps = cam_fps
            state.last_error = None
            frame_ready.notify_all()


def _inference_loop(
    model_name: str,
    imgsz: int,
    conf: float,
    infer_every: int,
    max_det: int,
) -> None:
    torch, YOLO = require_inference_stack()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO model:", model_name)
    print("YOLO device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    with state.lock:
        state.model_name = model_name

    model = YOLO(model_name, task="detect")
    if model_name.lower().endswith(".pt"):
        model.to(device)
        try:
            model.fuse()
        except Exception as exc:
            print(f"YOLO fuse skipped: {exc}")
    else:
        print("Skipping PyTorch-only model setup for exported inference model")

    warmup_started = time.perf_counter()
    warmup_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    print("Warming up YOLO inference...")
    with torch.inference_mode():
        model.predict(
            source=warmup_frame,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            device=device,
            max_det=max_det,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    with state.lock:
        state.inference_status = "ready"
        state.inference_warmup_seconds = warmup_seconds
    print(f"YOLO inference ready after {warmup_seconds:.2f}s")

    last_processed_frame_id = 0
    last_yolo_time = time.time()
    yolo_since_fps = 0

    while True:
        with frame_ready:
            frame_ready.wait_for(
                lambda: (
                    state.raw_frame is not None
                    and state.frame_count - last_processed_frame_id >= max(infer_every, 1)
                ),
                timeout=0.2,
            )
            if state.raw_frame is None or state.frame_count == last_processed_frame_id:
                continue
            frame = state.raw_frame.copy()
            frame_number = state.frame_count
            frame_received_monotonic = state.raw_frame_received_monotonic

        inference_started_monotonic = time.monotonic()
        inference_timestamp = time.time()
        with torch.inference_mode():
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=device,
                max_det=max_det,
            )

        detections: list[dict[str, Any]] = []
        tracks: list[dict[str, Any]] = []

        if results and results[0].boxes is not None:
            names = results[0].names
            boxes = results[0].boxes

            for box in boxes:
                detections.append(make_detection(box, names, inference_timestamp, frame_number))

        tracks = tracker.update(detections, frame_received_monotonic)
        current_tracks = current_detection_tracks(tracks)
        inference_completed_monotonic = time.monotonic()
        processed = ProcessedFrameBundle(
            frame_id=frame_number,
            frame_receipt_monotonic_s=frame_received_monotonic,
            frame_shape=tuple(int(value) for value in frame.shape),
            inference_started_monotonic_s=inference_started_monotonic,
            inference_completed_monotonic_s=inference_completed_monotonic,
            detections=tuple(dict(detection) for detection in detections),
            tracks=current_tracks,
        )
        last_processed_frame_id = frame_number
        yolo_since_fps += 1

        now = time.time()
        yolo_fps = None
        if now - last_yolo_time >= 1.0:
            yolo_fps = yolo_since_fps / max(now - last_yolo_time, 1e-6)
            yolo_since_fps = 0
            last_yolo_time = now

        with frame_ready:
            if not processed_frame_is_newer(state.processed_frame, processed):
                continue
            state.latest_detections = detections
            state.latest_tracks = tracks
            state.processed_frame = processed
            state.tabletop_localization = tabletop_localization(list(current_tracks))
            state.yolo_count += 1
            if yolo_fps is not None:
                state.yolo_fps = yolo_fps
            frame_ready.notify_all()


def _tabletop_homography(value: dict[str, Any]) -> np.ndarray | None:
    frame = value.get("camera_frame") or {}
    tabletop = value.get("tabletop") or {}
    corners = value.get("corners_px") or []
    if len(corners) != 4:
        return None
    try:
        width = float(tabletop["width_m"])
        depth = float(tabletop["depth_m"])
        src = np.array([[float(c["x"]), float(c["y"])] for c in corners], dtype=float)
        dst = np.array([[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not frame.get("width") or not frame.get("height") or width <= 0 or depth <= 0:
        return None
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (u, v), (x, y) in zip(src, dst):
        rows.extend(
            [[u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x], [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y]]
        )
        rhs.extend([x, y])
    try:
        h = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
    except np.linalg.LinAlgError:
        return None
    return np.r_[h, 1.0].reshape(3, 3)


def tabletop_localization(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    global _tabletop_cache_mtime, _tabletop_cache
    try:
        mtime = TABLETOP_CALIBRATION_PATH.stat().st_mtime
    except OSError:
        return {"configured": False, "status": "not_configured"}
    with _tabletop_cache_lock:
        if _tabletop_cache is None or _tabletop_cache_mtime != mtime:
            try:
                _tabletop_cache = json.loads(TABLETOP_CALIBRATION_PATH.read_text(encoding="utf-8"))
                _tabletop_cache_mtime = mtime
            except (OSError, json.JSONDecodeError):
                return {"configured": False, "status": "invalid_map"}
        value = _tabletop_cache
    homography = _tabletop_homography(value)
    if homography is None:
        return {"configured": False, "status": "invalid_map"}
    target = max(tracks, key=lambda item: float(item.get("confidence", 0.0)), default=None)
    if target is None or not isinstance(target.get("center_xy"), list):
        return {"configured": True, "status": "waiting_for_track"}
    u, v = (float(target["center_xy"][0]), float(target["center_xy"][1]))
    projected = homography @ np.array([u, v, 1.0], dtype=float)
    if abs(projected[2]) < 1e-9:
        return {
            "configured": True,
            "status": "projection_failed",
            "track_id": target.get("track_id"),
        }
    x, y = (float(projected[0] / projected[2]), float(projected[1] / projected[2]))
    tabletop = value["tabletop"]
    inside = 0.0 <= x <= float(tabletop["width_m"]) and 0.0 <= y <= float(tabletop["depth_m"])
    return {
        "configured": True,
        "status": "localized" if inside else "outside_tabletop",
        "track_id": target.get("track_id"),
        "pixel_xy": [round(u, 2), round(v, 2)],
        "table_xy_m": [round(x, 4), round(y, 4)],
        "inside": inside,
    }


def inference_loop(
    model_name: str,
    imgsz: int,
    conf: float,
    infer_every: int,
    max_det: int,
) -> None:
    with state.lock:
        state.inference_status = "warming_up"
        state.inference_error = None

    try:
        _inference_loop(model_name, imgsz, conf, infer_every, max_det)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with state.lock:
            state.inference_status = "error"
            state.inference_error = error
        print(f"ERROR: YOLO inference worker failed: {error}")
        traceback.print_exc()


def jpeg_loop(jpeg_quality: int) -> None:
    last_encoded_frame_id = 0
    last_encode_time = time.time()
    encodes_since_fps = 0

    while True:
        with frame_ready:
            frame_ready.wait_for(
                lambda: state.raw_frame is not None and state.frame_count != last_encoded_frame_id,
                timeout=0.2,
            )
            if state.raw_frame is None or state.frame_count == last_encoded_frame_id:
                continue
            frame_number = state.frame_count
            frame = state.raw_frame.copy()
            tracks = [dict(track) for track in state.latest_tracks]
            cam_fps = state.fps
            yolo_fps = state.yolo_fps

        display = frame
        draw_tracks(display, tracks)
        draw_fps(display, cam_fps, yolo_fps)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ok_jpeg, jpeg = cv2.imencode(".jpg", display, encode_params)

        if ok_jpeg:
            last_encoded_frame_id = frame_number
            encodes_since_fps += 1
            now = time.time()
            encode_fps = None
            if now - last_encode_time >= 1.0:
                encode_fps = encodes_since_fps / max(now - last_encode_time, 1e-6)
                encodes_since_fps = 0
                last_encode_time = now

            with jpeg_ready:
                state.annotated_frame = display
                state.jpeg_bytes = jpeg.tobytes()
                state.jpeg_frame_id = frame_number
                state.encode_count += 1
                if encode_fps is not None:
                    state.encode_fps = encode_fps
                state.last_error = None
                jpeg_ready.notify_all()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(GB10_WEB_DIR / "unitree_dual_viewer.html")


@app.get("/unitree_dual_viewer.html")
def viewer() -> FileResponse:
    return FileResponse(GB10_WEB_DIR / "unitree_dual_viewer.html")


@app.get("/tabletop-calibration")
def tabletop_calibration_page() -> FileResponse:
    return FileResponse(GB10_WEB_DIR / "tabletop_calibration.html")


def _commissioning_html() -> str:
    """Serve the existing guarded wizard without the retired bearer-token UI."""

    html = COMMISSIONING_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(
        '<div class="field"><label for="token">Bearer token — held in memory only</label><input id="token" type="password" autocomplete="off" /></div>',
        "",
    )
    html = html.replace(
        'let state = null, sessionId = null, sequence = 0, pollTimer = null, heartbeatTimer = null, authToken = "";',
        "let state = null, sessionId = null, sequence = 0, pollTimer = null, heartbeatTimer = null;",
    )
    html = html.replace(
        'const headers = () => ({"Authorization": `Bearer ${authToken}`, "Content-Type": "application/json"});',
        'const headers = () => ({"Content-Type": "application/json"});',
    )
    html = html.replace(
        "Enter the token and connect. No movement occurs on connection.",
        "Connect to inspect state. No movement occurs on connection.",
    )
    html = html.replace(
        'authToken=$("token").value; await refresh(); if (state) $("token").value="";',
        "await refresh();",
    )
    return html


@app.get("/commissioning")
def commissioning_redirect() -> RedirectResponse:
    return RedirectResponse("/commissioning/", status_code=307)


@app.get("/commissioning/", response_class=HTMLResponse)
def commissioning_page() -> str:
    return _commissioning_html()


def _require_transport() -> Any:
    if tracking_transport is None:
        raise HTTPException(status_code=503, detail="ROS 2 control transport is unavailable")
    return tracking_transport


def _transport_call(operation: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return dict(call())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{operation} failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/arm/state")
def arm_state_api() -> dict[str, Any]:
    transport = _require_transport()
    return _transport_call("arm state", transport.arm_state)


@app.post("/api/v1/arm/follow-profile")
def arm_follow_profile_api(payload: dict[str, Any]) -> dict[str, Any]:
    if arm_runtime is None:
        raise HTTPException(status_code=503, detail="arm tracking runtime is unavailable")
    profile = str(payload.get("follow_profile") or "")
    if profile not in {"balanced", "aggressive"}:
        raise HTTPException(
            status_code=422,
            detail="follow_profile must be balanced or aggressive",
        )
    try:
        return arm_runtime.set_follow_profile(profile)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/arm/enable")
def arm_enable_api(payload: dict[str, Any]) -> dict[str, Any]:
    transport = _require_transport()
    session_id = str(payload.get("session_id") or "")
    calibration_id = str(payload.get("calibration_id") or "")
    if not session_id or not calibration_id:
        raise HTTPException(status_code=422, detail="session_id and calibration_id are required")
    if arm_runtime is not None:
        requested_profile = payload.get("follow_profile")
        if requested_profile is not None:
            profile = str(requested_profile)
            if profile not in {"balanced", "aggressive"}:
                raise HTTPException(
                    status_code=422,
                    detail="follow_profile must be balanced or aggressive",
                )
            try:
                arm_runtime.set_follow_profile(profile)
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        # Release any prior latched interception before enabling the new
        # session. Resetting after enable could immediately stop the session
        # that the operator just created.
        arm_runtime.reset_intercept()
    report = _transport_call(
        "arm enable",
        lambda: transport.enable_arm(session_id, calibration_id),
    )
    return report


@app.post("/api/v1/arm/stop")
def arm_stop_api(payload: dict[str, Any]) -> dict[str, Any]:
    transport = _require_transport()
    reason = str(payload.get("reason") or "operator_stop")

    def stop() -> dict[str, Any]:
        transport.stop_arm(reason)
        return {"ok": True, "reason": reason}

    return _transport_call("arm stop", stop)


@app.post("/api/v1/arm/return-to-neutral")
def arm_return_api(payload: dict[str, Any]) -> dict[str, Any]:
    transport = _require_transport()
    reason = str(payload.get("reason") or "operator_return")
    return _transport_call(
        "arm return",
        lambda: transport.return_arm(reason),
    )


@app.post("/api/v1/intercept/reset")
def intercept_reset_api() -> dict[str, Any]:
    if arm_runtime is None:
        raise HTTPException(status_code=503, detail="arm tracking runtime is unavailable")
    arm_runtime.reset_intercept()
    return {"ok": True, "state": "ACQUIRING"}


@app.get("/api/v1/commissioning/state")
def commissioning_state_api() -> dict[str, Any]:
    transport = _require_transport()
    return _transport_call(
        "commissioning state",
        lambda: transport.commissioning("state", {}),
    )


@app.post("/api/v1/commissioning/sessions")
def commissioning_session_api(payload: dict[str, Any]) -> dict[str, Any]:
    transport = _require_transport()
    return _transport_call(
        "commissioning session",
        lambda: transport.commissioning("create_session", dict(payload)),
    )


_COMMISSIONING_ACTIONS = {
    "enable": "enable",
    "heartbeat": "heartbeat",
    "jogs": "jog",
    "confirmations": "confirm",
    "checkpoints": "checkpoint",
    "candidate": "capture_candidate",
    "replay-steps": "replay_step",
    "replay-validations": "validate_replay",
    "stop": "stop",
    "promote": "promote",
}


@app.post("/api/v1/commissioning/sessions/{session_id}/{action}")
def commissioning_action_api(
    session_id: str,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    command = _COMMISSIONING_ACTIONS.get(action)
    if command is None:
        raise HTTPException(status_code=404, detail="unknown commissioning action")
    transport = _require_transport()
    request_payload = {**payload, "session_id": session_id}
    return _transport_call(
        f"commissioning {command}",
        lambda: transport.commissioning(command, request_payload),
    )


@app.get("/health")
def health() -> dict[str, object]:
    with state.lock:
        fps_target_met = state.expected_fps <= 0 or (
            state.fps > 0 and state.fps >= state.expected_fps * 0.9
        )
        return {
            "ok": (
                state.jpeg_bytes is not None
                and state.inference_status in {"disabled", "ready"}
                and fps_target_met
            ),
            "frame_count": state.frame_count,
            "jpeg_frame_id": state.jpeg_frame_id,
            "yolo_count": state.yolo_count,
            "encode_count": state.encode_count,
            "fps": state.fps,
            "yolo_fps": state.yolo_fps,
            "encode_fps": state.encode_fps,
            "last_error": state.last_error,
            "opencv": cv2.__version__,
            "opencv_gstreamer": opencv_gstreamer_enabled(),
            "capture_backend": state.capture_backend,
            "camera_name": state.camera_name,
            "camera_pipeline": state.camera_pipeline,
            "torch_cuda": torch_cuda_available(),
            "stream_fps_limit": state.stream_fps_limit,
            "expected_fps": state.expected_fps,
            "fps_target_met": fps_target_met,
            "model_name": state.model_name,
            "inference_status": state.inference_status,
            "inference_error": state.inference_error,
            "inference_warmup_seconds": state.inference_warmup_seconds,
            "tabletop_localization": dict(state.tabletop_localization),
            "opencv_threads": state.opencv_threads,
            "torch_threads": state.torch_threads,
            "arm_tracking": dict(state.arm_tracking),
            "research": (
                {"enabled": False}
                if research_session is None
                else research_session.session_report()
            ),
        }


_TABLETOP_CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")


def _validate_tabletop_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        width_m = float(payload["width_m"])
        depth_m = float(payload["depth_m"])
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
        corners = payload["corners"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid tabletop calibration payload") from exc
    if not (0.05 <= width_m <= 5.0 and 0.05 <= depth_m <= 5.0):
        raise HTTPException(
            status_code=422, detail="tabletop dimensions must be between 5 cm and 5 m"
        )
    if image_width <= 0 or image_height <= 0:
        raise HTTPException(status_code=422, detail="camera frame dimensions must be positive")
    if not isinstance(corners, list) or len(corners) != len(_TABLETOP_CORNER_ORDER):
        raise HTTPException(status_code=422, detail="exactly four tabletop corners are required")
    validated: list[dict[str, Any]] = []
    for expected, item in zip(_TABLETOP_CORNER_ORDER, corners):
        if not isinstance(item, dict) or item.get("name") != expected:
            raise HTTPException(
                status_code=422, detail="corners must be near-left, near-right, far-right, far-left"
            )
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid {expected} corner") from exc
        if not (0.0 <= x <= image_width and 0.0 <= y <= image_height):
            raise HTTPException(status_code=422, detail=f"{expected} is outside the camera frame")
        validated.append({"name": expected, "x": round(x, 3), "y": round(y, 3)})
    return {
        "schema_version": 1,
        "camera_frame": {"width": image_width, "height": image_height},
        "tabletop": {"width_m": width_m, "depth_m": depth_m},
        "corner_order": list(_TABLETOP_CORNER_ORDER),
        "corners_px": validated,
    }


@app.get("/api/v1/tabletop-calibration")
def tabletop_calibration_get() -> dict[str, Any]:
    if not TABLETOP_CALIBRATION_PATH.is_file():
        return {"configured": False}
    try:
        value = json.loads(TABLETOP_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"could not read tabletop calibration: {exc}"
        ) from exc
    return {"configured": True, "calibration": value}


@app.post("/api/v1/tabletop-calibration")
def tabletop_calibration_save(payload: dict[str, Any]) -> dict[str, Any]:
    value = _validate_tabletop_calibration(payload)
    value["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        TABLETOP_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = TABLETOP_CALIBRATION_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(TABLETOP_CALIBRATION_PATH)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"could not save tabletop calibration: {exc}"
        ) from exc
    return {"ok": True, "path": str(TABLETOP_CALIBRATION_PATH), "calibration": value}


@app.get("/snapshot.jpg")
def snapshot() -> Response:
    with state.lock:
        data = state.jpeg_bytes

    if data is None:
        return Response(content=b"No frame yet", media_type="text/plain", status_code=503)

    return Response(content=data, media_type="image/jpeg")


@app.get("/raw-snapshot.jpg")
def raw_snapshot() -> Response:
    """Return one clean latest RGB observation for policy inference.

    Unlike ``/snapshot.jpg``, this endpoint contains no boxes, tracks, or FPS
    text that would shift the VLA input distribution.  Encoding happens only
    when requested by an operator/policy process, not in the camera hot path.
    """

    with state.lock:
        frame = None if state.raw_frame is None else state.raw_frame.copy()
        received_at = state.raw_frame_received_monotonic
        frame_id = state.frame_count
    if frame is None:
        return Response(content=b"No frame yet", media_type="text/plain", status_code=503)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return Response(
            content=b"Could not encode latest frame",
            media_type="text/plain",
            status_code=503,
        )
    age_ms = max(0.0, (time.monotonic() - received_at) * 1000.0)
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-G1-Frame-Id": str(frame_id),
            "X-G1-Frame-Age-Ms": f"{age_ms:.3f}",
            "Cache-Control": "no-store",
        },
    )


@app.get("/detections")
def detections() -> dict[str, Any]:
    with state.lock:
        return {
            "frame_count": state.frame_count,
            "yolo_count": state.yolo_count,
            "detections": state.latest_detections,
        }


@app.get("/tracks")
def tracks() -> dict[str, Any]:
    with state.lock:
        return {
            "frame_count": state.frame_count,
            "yolo_count": state.yolo_count,
            "tracks": state.latest_tracks,
        }


@app.get("/arm-tracking")
def arm_tracking() -> dict[str, Any]:
    with state.lock:
        return dict(state.arm_tracking)


@app.get("/api/v1/dashboard")
def dashboard_state() -> dict[str, Any]:
    """Return only the newest fields needed by the browser overlay.

    This avoids serializing the complete health/visualization tree twice every
    refresh and keeps overlay rendering entirely off the camera/YOLO threads.
    """

    with state.lock:
        tracking = state.arm_tracking
        visual = tracking.get("visualization")
        if not isinstance(visual, dict):
            visual = {}
        return {
            "camera": {
                "frame_id": state.jpeg_frame_id,
                "fps": state.fps,
                "yolo_fps": state.yolo_fps,
                "encode_fps": state.encode_fps,
                "inference": state.inference_status,
                "width": (
                    None if state.raw_frame is None else int(state.raw_frame.shape[1])
                ),
                "height": (
                    None if state.raw_frame is None else int(state.raw_frame.shape[0])
                ),
            },
            "tracking": {
                key: tracking.get(key)
                for key in (
                    "status",
                    "reason",
                    "arm_state",
                    "arm_weight",
                    "detector_confidence",
                    "depth_age_ms",
                    "pair_skew_ms",
                    "processing_latency_ms",
                    "pipeline_age_ms",
                    "ik_latency_ms",
                    "ik_step_type",
                    "target_xyz_m",
                    "object_xyz_m",
                )
            },
            "visualization": {
                key: visual.get(key)
                for key in (
                    "camera",
                    "support_plane",
                    "support_plane_status",
                    "measured_object_xyz_m",
                    "predicted_object_xyz_m",
                    "predicted_trajectory_xyz_m",
                    "pregrasp_target_xyz_m",
                    "end_effector_xyz_m",
                )
            },
        }


@app.get("/visualization/state")
def visualization_state_report() -> dict[str, Any]:
    """Read-only scene state; this endpoint has no mutation counterpart."""

    with state.lock:
        visual = state.arm_tracking.get("visualization")
        if isinstance(visual, dict):
            return dict(visual)
    from object_tracking.arm_tracking.visualization import visualization_state

    return visualization_state(measured_body_q=None, available=False)


@app.get("/depth.jpg")
def depth_colormap() -> Response:
    with state.lock:
        data = state.depth_colormap_jpeg
    if data is None:
        return Response(content=b"No depth frame yet", media_type="text/plain", status_code=503)
    return Response(content=data, media_type="image/jpeg")


@app.get("/research/session")
def research_session_report() -> dict[str, Any]:
    if research_session is None:
        return {"enabled": False, "reason": "research recording disabled"}
    return research_session.session_report()


@app.get("/research/summary")
def research_summary() -> dict[str, Any]:
    if research_session is None:
        return {"enabled": False, "reason": "research recording disabled"}
    return {"enabled": True, **research_session.summary_report()}


@app.get("/research/telemetry")
def research_telemetry(limit: int = 100) -> dict[str, Any]:
    if research_session is None:
        return {"enabled": False, "samples": []}
    return {
        "enabled": True,
        "session_id": research_session.session_id,
        "samples": research_session.recent_samples(limit),
    }


@app.get("/research/export.jsonl")
def research_export() -> Response:
    if research_session is None or not research_session.telemetry_path.exists():
        return Response(
            content=b"Research recording is disabled or has no samples yet",
            media_type="text/plain",
            status_code=404,
        )
    return FileResponse(
        research_session.telemetry_path,
        media_type="application/x-ndjson",
        filename=f"{research_session.session_id}_telemetry.jsonl",
    )


def _capture_session_dir() -> Path:
    with state.lock:
        if state.capture_session_dir is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.capture_session_dir = Path("runs/captures/plushie") / session_name
        return state.capture_session_dir


@app.get("/capture")
@app.post("/capture")
def capture(note: str = "") -> dict[str, Any]:
    session_dir = _capture_session_dir()
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with state.lock:
        if state.raw_frame is None:
            return {
                "ok": False,
                "error": "No raw frame available yet",
            }

        state.capture_count += 1
        capture_count = state.capture_count
        frame_number = state.frame_count
        pipeline = state.camera_pipeline
        camera_name = state.camera_name
        frame = state.raw_frame.copy()

    frame_name = f"frame_{capture_count:06d}.jpg"
    frame_path = images_dir / frame_name
    ok = cv2.imwrite(str(frame_path), frame)
    if not ok:
        return {
            "ok": False,
            "error": f"Could not write {frame_path}",
        }

    metadata = {
        "timestamp": time.time(),
        "source": "unitree_g1",
        "camera_name": camera_name,
        "frame_number": frame_number,
        "frame_path": str(Path("images") / frame_name),
        "camera_pipeline": pipeline,
        "note": note,
    }

    metadata_path = session_dir / "metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")

    return {
        "ok": True,
        "session_dir": str(session_dir),
        "frame_path": str(frame_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def mjpeg_generator():
    last_sent_frame = -1

    while True:
        with jpeg_ready:
            jpeg_ready.wait_for(
                lambda: state.jpeg_bytes is not None and state.jpeg_frame_id != last_sent_frame,
                timeout=1.0,
            )
            data = state.jpeg_bytes
            frame_count = state.jpeg_frame_id
            stream_fps_limit = state.stream_fps_limit

        if data is None or frame_count == last_sent_frame:
            continue

        last_sent_frame = frame_count

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" + data + b"\r\n"
        )

        if stream_fps_limit > 0:
            time.sleep(1.0 / stream_fps_limit)


@app.get("/stream.mjpg")
def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a low-latency MJPEG stream from a GStreamer camera source."
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument(
        "--model",
        default="none",
        help="YOLO model path/name. Use none to disable detector dependencies.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--opencv-threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--torch-threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Print one Uvicorn line for every browser/API request (off by default).",
    )
    parser.add_argument(
        "--stream-fps",
        type=float,
        default=0.0,
        help="Maximum MJPEG response FPS. Use 0 for unbounded/latest-frame streaming.",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=0.0,
        help="Expected source FPS for health readiness. Allows a 10 percent tolerance; use 0 to disable.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "opencv", "gst-launch"),
        default="auto",
        help="Capture backend to use for camera read. opencv requires GStreamer-enabled cv2 and does not fallback.",
    )
    parser.add_argument(
        "--calibration",
        help="Validated camera-to-torso calibration YAML; enables ROS 2 depth tracking",
    )
    parser.add_argument("--arm-home", help="Validated commissioned right-arm home profile")
    parser.add_argument("--robot-id", help="Robot identity bound to the commissioned arm home")
    parser.add_argument(
        "--allow-nominal-support-plane",
        action="store_true",
        help=(
            "When the physical table is absent, use the calibration-derived "
            "torso-frame exclusion plane while retaining all other motion gates"
        ),
    )
    parser.add_argument("--target-hz", type=float, default=15.0)
    parser.add_argument(
        "--follow-profile",
        choices=("balanced", "aggressive"),
        default="balanced",
        help="Realtime arm-follow responsiveness profile",
    )
    parser.add_argument(
        "--ros-depth-only",
        action="store_true",
        help="Subscribe only to /g1/depth; manual arm control uses its separate ROS client",
    )
    parser.add_argument(
        "--ros-observe-only",
        action="store_true",
        help="Subscribe to depth and arm state without creating any command publishers",
    )
    parser.add_argument(
        "--trajectory-model",
        help="Optional trained 3D trajectory checkpoint; invalid/insufficient history falls back to alpha-beta",
    )
    parser.add_argument(
        "--intercept-config",
        help=(
            "Versioned, calibration-bound fixed-lane interception profile. "
            "Omit to preserve continuous tracking mode; movement still requires "
            "robot-side permission and --execute."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Publish arm targets only after every calibration and health gate passes",
    )
    parser.add_argument(
        "--research-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record structured telemetry under runs/research/arm_tracking (default: enabled)",
    )
    parser.add_argument(
        "--research-root",
        default="runs/research/arm_tracking",
        help="Root directory for isolated research sessions",
    )
    parser.add_argument(
        "--research-hz",
        type=float,
        default=15.0,
        help="Structured telemetry sampling rate (default: 15 Hz)",
    )
    parser.add_argument("--research-label", default="", help="Human-readable experiment label")
    parser.add_argument("--research-notes", default="", help="Short experimental condition notes")
    return parser


def assert_port_available(host: str, port: int) -> None:
    bind_host = "" if host in {"", "0.0.0.0"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise SystemExit(
                f"Port {port} is already in use. Stop the old stream server or run with PORT={port + 1}."
            ) from exc


def arm_tracking_snapshot() -> dict[str, Any]:
    with state.lock:
        processed = state.processed_frame
        if processed is None:
            return {
                "rgb_receipt_time_s": 0.0,
                "rgb_shape": None,
                "rgb_frame_id": None,
                "inference_started_monotonic_s": None,
                "inference_completed_monotonic_s": None,
                "tracks": [],
            }
        return {
            "rgb_receipt_time_s": processed.frame_receipt_monotonic_s,
            "rgb_shape": processed.frame_shape,
            "rgb_frame_id": processed.frame_id,
            "inference_started_monotonic_s": processed.inference_started_monotonic_s,
            "inference_completed_monotonic_s": processed.inference_completed_monotonic_s,
            "tracks": [dict(track) for track in processed.tracks],
        }


def update_arm_tracking(status: dict[str, Any], depth_jpeg: bytes | None) -> None:
    with state.lock:
        state.arm_tracking = dict(status)
        if depth_jpeg is not None:
            state.depth_colormap_jpeg = depth_jpeg


def raw_depth_preview_loop(transport: Any, trajectory_model: str | None) -> None:
    """Render the robot's Z16 topic without inventing torso-frame localization.

    This is intentionally an observe-only consumer. It reports factory D435
    colour registration separately from the still-required camera-to-torso
    calibration, and never fabricates a robot-frame position.
    """

    from object_tracking.arm_tracking.runtime import depth_colormap_jpeg

    update_arm_tracking(
        {
            "enabled": False,
            "mode": "observe-only",
            "status": "waiting_for_depth",
            "reason": "Raw depth preview is waiting on ROS /g1/depth; 3D prediction requires calibration.",
            "trajectory_model": trajectory_model,
            "prediction_source": None,
        },
        None,
    )
    last_frame_at = time.monotonic()
    while not depth_preview_stop.is_set():
        try:
            frame = transport.receive_depth(timeout_s=0.5)
            if frame is None:
                if time.monotonic() - last_frame_at >= 1.0:
                    update_arm_tracking(
                        {
                            "enabled": False,
                            "mode": "observe-only",
                            "status": "waiting_for_depth",
                            "reason": "ROS /g1/depth stopped publishing; retaining the last preview frame.",
                            "trajectory_model": trajectory_model,
                            "prediction_source": None,
                        },
                        None,
                    )
                continue
            last_frame_at = time.monotonic()
            age_ms = max(0.0, (time.monotonic() - frame.receipt_time_s) * 1000.0)
            if frame.registered_to_rgb:
                status = "aligned_depth_preview"
                reason = (
                    "D435 RGB-aligned depth is live. Pixel-level 3D is ready; "
                    "camera-to-torso calibration is still required before IK or prediction."
                )
            else:
                status = "depth_preview"
                reason = (
                    "Raw Z16 depth is live. It is not RGB-registered, so the 3D "
                    "position predictor is deliberately inactive until calibration."
                )
            update_arm_tracking(
                {
                    "enabled": False,
                    "mode": "observe-only",
                    "status": status,
                    "reason": reason,
                    "depth_sequence": frame.sequence,
                    "depth_age_ms": round(age_ms, 3),
                    "depth_scale_m_per_unit": frame.depth_scale,
                    "depth_profile": [int(frame.z16.shape[1]), int(frame.z16.shape[0])],
                    "depth_registered_to_rgb": frame.registered_to_rgb,
                    "calibration_id": frame.calibration_id,
                    "trajectory_model": trajectory_model,
                    "prediction_source": None,
                },
                depth_colormap_jpeg(frame.z16, frame.depth_scale),
            )
        except Exception as exc:
            update_arm_tracking(
                {
                    "enabled": False,
                    "mode": "observe-only",
                    "status": "waiting_for_depth",
                    "reason": f"depth_preview_error: {type(exc).__name__}: {exc}",
                    "trajectory_model": trajectory_model,
                    "prediction_source": None,
                },
                None,
            )
            depth_preview_stop.wait(0.25)


def start_raw_depth_preview(transport: Any, trajectory_model: str | None) -> None:
    global depth_preview_thread
    if depth_preview_thread is not None:
        return
    depth_preview_stop.clear()
    depth_preview_thread = threading.Thread(
        target=raw_depth_preview_loop,
        args=(transport, trajectory_model),
        daemon=True,
        name="raw-depth-preview",
    )
    depth_preview_thread.start()


def research_recording_loop(sample_hz: float) -> None:
    period = 1.0 / sample_hz
    while True:
        started = time.monotonic()
        with state.lock:
            sample = {
                "frame_count": state.frame_count,
                "yolo_count": state.yolo_count,
                "encode_count": state.encode_count,
                "fps": state.fps,
                "yolo_fps": state.yolo_fps,
                "encode_fps": state.encode_fps,
                "inference_status": state.inference_status,
                "inference_error": state.inference_error,
                "camera_name": state.camera_name,
                "model_name": state.model_name,
                "detections": [dict(item) for item in state.latest_detections],
                "tracks": [dict(item) for item in state.latest_tracks],
                "arm_tracking": dict(state.arm_tracking),
            }
        if research_session is not None:
            research_session.record(sample)
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, period - elapsed))


def shutdown_ros_transport() -> None:
    global arm_runtime, tracking_transport, depth_preview_thread
    depth_preview_stop.set()
    if depth_preview_thread is not None:
        depth_preview_thread.join(timeout=2.0)
        depth_preview_thread = None
    if arm_runtime is not None:
        arm_runtime.stop()
        arm_runtime = None
    elif tracking_transport is not None:
        try:
            tracking_transport.close()
        except Exception:
            pass
    tracking_transport = None


def main() -> None:
    global arm_runtime, research_session, tracking_transport
    args = build_parser().parse_args()
    assert_port_available(args.host, args.port)
    detector_enabled = yolo_enabled(args.model)
    if detector_enabled and not Path(args.model).is_file():
        raise SystemExit(
            f"YOLO checkpoint not found: {args.model}. Provision the ignored model artifact before startup."
        )
    if args.execute and not args.calibration:
        raise SystemExit("--execute requires --calibration")
    if args.intercept_config and not args.calibration:
        raise SystemExit("--intercept-config requires --calibration")
    if args.research_hz <= 0 or args.research_hz > 30:
        raise SystemExit("--research-hz must be greater than 0 and at most 30")
    configure_runtime(args.opencv_threads, args.torch_threads, detector_enabled)

    with state.lock:
        state.stream_fps_limit = max(args.stream_fps, 0.0)
        state.expected_fps = max(args.expected_fps, 0.0)
        state.inference_status = "warming_up" if detector_enabled else "disabled"
        state.inference_error = None

    if args.research_record:
        research_session = ResearchSession(
            args.research_root,
            metadata={
                "mode": "execute" if args.execute else "dry-run",
                "camera_name": args.camera_name,
                "model": args.model,
                "expected_fps": args.expected_fps,
                "target_hz": args.target_hz,
                "control_transport": "ros2",
                "calibration_file": args.calibration,
                "intercept_config_file": args.intercept_config,
                "arm_home_file": args.arm_home,
                "robot_id": args.robot_id,
                "experiment_label": args.research_label,
                "experiment_notes": args.research_notes,
                "configuration": {
                    "imgsz": args.imgsz,
                    "confidence": args.conf,
                    "infer_every": args.infer_every,
                    "jpeg_quality": args.jpeg_quality,
                    "max_detections": args.max_det,
                    "opencv_threads": args.opencv_threads,
                    "torch_threads": args.torch_threads,
                    "stream_fps": args.stream_fps,
                    "expected_fps": args.expected_fps,
                    "capture_backend": args.capture_backend,
                    "target_hz": args.target_hz,
                    "research_hz": args.research_hz,
                    "intercept_config": args.intercept_config,
                },
            },
        )
        print(f"Research session: {research_session.session_id}")
        print(f"Research data:    {research_session.directory}")
        research_worker = threading.Thread(
            target=research_recording_loop,
            args=(args.research_hz,),
            daemon=True,
            name="research-recorder",
        )
        research_worker.start()

    camera_worker = threading.Thread(
        target=camera_loop,
        args=(args.pipeline, args.camera_name, args.capture_backend),
        daemon=True,
    )
    jpeg_worker = threading.Thread(
        target=jpeg_loop,
        args=(args.jpeg_quality,),
        daemon=True,
    )
    camera_worker.start()
    try:
        from object_tracking.ros2_tracking import create_ros_tracking_transport

        tracking_transport = create_ros_tracking_transport(
            observe_depth_only=args.ros_depth_only,
            observe_only=args.ros_observe_only,
        )
        if args.calibration:
            from object_tracking.arm_tracking.runtime import (
                ArmTrackingRuntime,
                RuntimeConfig,
                follow_profile_limits,
            )

            (
                maximum_joint_step_rad,
                maximum_velocity_rad_s,
                maximum_acceleration_rad_s2,
                maximum_jerk_rad_s3,
            ) = follow_profile_limits(args.follow_profile)

            arm_runtime = ArmTrackingRuntime(
                RuntimeConfig(
                    calibration_path=Path(args.calibration),
                    tabletop_path=TABLETOP_CALIBRATION_PATH,
                    allow_nominal_support_plane=args.allow_nominal_support_plane,
                    arm_home_path=None if args.arm_home is None else Path(args.arm_home),
                    robot_id=args.robot_id,
                    execute=args.execute,
                    target_hz=args.target_hz,
                    follow_profile=args.follow_profile,
                    maximum_joint_step_rad=maximum_joint_step_rad,
                    maximum_velocity_rad_s=maximum_velocity_rad_s,
                    maximum_acceleration_rad_s2=maximum_acceleration_rad_s2,
                    maximum_jerk_rad_s3=maximum_jerk_rad_s3,
                    trajectory_model_path=(
                        None if args.trajectory_model is None else Path(args.trajectory_model)
                    ),
                    intercept_config_path=(
                        None if args.intercept_config is None else Path(args.intercept_config)
                    ),
                ),
                arm_tracking_snapshot,
                update_arm_tracking,
                transport=tracking_transport,
                repo_root=REPO_ROOT,
            )
            arm_runtime.start()
        else:
            tracking_transport.start()
            start_raw_depth_preview(tracking_transport, args.trajectory_model)
    except Exception as exc:
        if tracking_transport is not None:
            try:
                tracking_transport.close()
            except Exception:
                pass
        tracking_transport = None
        update_arm_tracking(
            {
                "enabled": bool(args.calibration),
                "mode": "execute" if args.execute else "dry-run",
                "status": "error",
                "reason": f"ROS 2 transport unavailable: {type(exc).__name__}: {exc}",
            },
            None,
        )
    if detector_enabled:
        inference_worker = threading.Thread(
            target=inference_loop,
            args=(
                args.model,
                args.imgsz,
                args.conf,
                args.infer_every,
                args.max_det,
            ),
            daemon=True,
        )
        inference_worker.start()
    else:
        print("YOLO detector disabled; serving camera frames only.")
    jpeg_worker.start()

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=args.access_log,
    )


if __name__ == "__main__":
    main()
