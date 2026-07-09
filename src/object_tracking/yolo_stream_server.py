from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from object_tracking.simple_tracker import SimpleTracker


def unitree_g1_videohub_pipeline(device: str, width: int = 1280, height: int = 720, out_width: int = 640, out_height: int = 360) -> str:
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
DEFAULT_PIPELINE = os.environ.get("G1_CAMERA_PIPELINE") or unitree_g1_videohub_pipeline(DEFAULT_CAMERA_DEVICE)


@dataclass
class SharedState:
    raw_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    jpeg_bytes: Optional[bytes] = None
    latest_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_tracks: list[dict[str, Any]] = field(default_factory=list)
    last_error: Optional[str] = None
    frame_count: int = 0
    jpeg_frame_id: int = 0
    yolo_count: int = 0
    encode_count: int = 0
    fps: float = 0.0
    yolo_fps: float = 0.0
    encode_fps: float = 0.0
    stream_fps_limit: float = 0.0
    camera_pipeline: str = DEFAULT_PIPELINE
    camera_name: str = DEFAULT_CAMERA_NAME
    capture_backend: str = ""
    model_name: str = "none"
    opencv_threads: int = 0
    torch_threads: int = 0
    capture_session_dir: Optional[Path] = None
    capture_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


state = SharedState()
frame_ready = threading.Condition(state.lock)
jpeg_ready = threading.Condition(state.lock)
tracker = SimpleTracker()
app = FastAPI()


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


def require_training_stack() -> tuple[Any, Any]:
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
            raise RuntimeError("OpenCV GStreamer backend requested, but cv2 was built without GStreamer.")
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
            state.frame_count += 1
            if cam_fps is not None:
                state.fps = cam_fps
            state.last_error = None
            frame_ready.notify_all()


def inference_loop(
    model_name: str,
    imgsz: int,
    conf: float,
    infer_every: int,
    max_det: int,
) -> None:
    torch, YOLO = require_training_stack()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO model:", model_name)
    print("YOLO device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    with state.lock:
        state.model_name = model_name

    model = YOLO(model_name)
    model.to(device)
    try:
        model.fuse()
    except Exception as exc:
        print(f"YOLO fuse skipped: {exc}")

    last_processed_frame_id = 0
    last_yolo_time = time.time()
    yolo_since_fps = 0

    while True:
        with frame_ready:
            frame_ready.wait_for(
                lambda: state.raw_frame is not None
                and state.frame_count - last_processed_frame_id >= max(infer_every, 1),
                timeout=0.2,
            )
            if state.raw_frame is None or state.frame_count == last_processed_frame_id:
                continue
            frame = state.raw_frame.copy()
            frame_number = state.frame_count

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

        tracks = tracker.update(detections, inference_timestamp)
        last_processed_frame_id = frame_number
        yolo_since_fps += 1

        now = time.time()
        yolo_fps = None
        if now - last_yolo_time >= 1.0:
            yolo_fps = yolo_since_fps / max(now - last_yolo_time, 1e-6)
            yolo_since_fps = 0
            last_yolo_time = now

        with frame_ready:
            state.latest_detections = detections
            state.latest_tracks = tracks
            state.yolo_count += 1
            if yolo_fps is not None:
                state.yolo_fps = yolo_fps
            frame_ready.notify_all()


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <html>
      <head>
        <title>Unitree G1 Vision Stream</title>
        <style>
          body { font-family: Arial, sans-serif; background: #111; color: #eee; }
          img { max-width: 100%; height: auto; border: 2px solid #444; }
          .wrap { max-width: 1100px; margin: 24px auto; }
          code { background: #222; padding: 2px 5px; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <h1>Unitree G1 Vision Stream</h1>
          <p>Camera stream from Ubuntu vision server.</p>
          <img src="/stream.mjpg" />
          <p>
            Snapshot: <a href="/snapshot.jpg">/snapshot.jpg</a><br/>
            Capture: <a href="/capture">/capture</a><br/>
            Detections: <a href="/detections">/detections</a><br/>
            Tracks: <a href="/tracks">/tracks</a><br/>
            Health: <a href="/health">/health</a>
          </p>
        </div>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, object]:
    with state.lock:
        return {
            "ok": state.jpeg_bytes is not None,
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
            "model_name": state.model_name,
            "opencv_threads": state.opencv_threads,
            "torch_threads": state.torch_threads,
        }


@app.get("/snapshot.jpg")
def snapshot() -> Response:
    with state.lock:
        data = state.jpeg_bytes

    if data is None:
        return Response(content=b"No frame yet", media_type="text/plain", status_code=503)

    return Response(content=data, media_type="image/jpeg")


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
            b"Cache-Control: no-cache\r\n\r\n"
            + data
            + b"\r\n"
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
    parser = argparse.ArgumentParser(description="Serve a low-latency MJPEG stream from a GStreamer camera source.")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--model", default="none", help="YOLO model path/name. Use none to disable detector dependencies.")
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
        "--stream-fps",
        type=float,
        default=0.0,
        help="Maximum MJPEG response FPS. Use 0 for unbounded/latest-frame streaming.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "opencv", "gst-launch"),
        default="auto",
        help="Capture backend to use for camera read. opencv requires GStreamer-enabled cv2 and does not fallback.",
    )
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


def main() -> None:
    args = build_parser().parse_args()
    assert_port_available(args.host, args.port)
    detector_enabled = yolo_enabled(args.model)
    configure_runtime(args.opencv_threads, args.torch_threads, detector_enabled)

    with state.lock:
        state.stream_fps_limit = max(args.stream_fps, 0.0)

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

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
