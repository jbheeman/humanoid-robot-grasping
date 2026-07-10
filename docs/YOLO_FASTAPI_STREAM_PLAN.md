# YOLO FastAPI Stream Implementation Plan

For efficiency, I would not use Streamlit as the actual video transport. Use a small headless Python vision server: OpenCV/GStreamer reads latest frame, YOLO draws boxes, FastAPI serves an MJPEG stream to your Mac.

Thought for 35s

Yes — this is the right next version:

Unitree camera feed
  → Ubuntu headless server
  → GStreamer/OpenCV reads latest frame
  → YOLOv8 detects objects
  → OpenCV draws bounding boxes
  → Ubuntu serves annotated video over a lightweight web endpoint
  → MacBook opens browser

I would not use Streamlit for this part. Streamlit is fine for dashboards, but for low-latency video preview, use a simple FastAPI MJPEG stream. FastAPI supports streaming responses, which is exactly what we need for a browser video feed.

Ultralytics YOLO works directly from Python and supports object detection/prediction use cases, so yolov8n.pt is a good lightweight substitute before the fine-tuned VLM comes later.

Target architecture
Robot / Unitree G1:
  publishes camera stream
  relays compressed H264 RTP to Ubuntu

Ubuntu headless server:
  receives UDP port 5600
  decodes with GStreamer through OpenCV
  runs YOLO every N frames
  draws boxes
  serves /stream.mjpg

MacBook:
  opens http://192.168.0.122:8000

Important: MacBook is only the viewer. The actual vision processing happens on Ubuntu.

1. Keep Unitree relay running

Run this on the Unitree robot:

```bash
gst-launch-1.0 -v \
  udpsrc multicast-group=230.1.1.1 address=0.0.0.0 port=1720 auto-multicast=true multicast-iface=wlan0 buffer-size=1048576 ! \
  "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000" ! \
  queue ! \
  udpsink host=192.168.0.122 port=5600 sync=false async=false
```

This sends compressed camera data to Ubuntu.

2. Create a dedicated system-Python vision venv

Run this on Ubuntu, not Unitree:

```bash
cd ~/Documents/project

uv venv --system-site-packages --allow-existing .venv
source .venv/bin/activate

uv pip install -e ".[vision-server]"
uv pip uninstall opencv-python opencv-contrib-python opencv-python-headless
```

The uninstall step matters because pip OpenCV usually has:

```text
GStreamer: NO
```

You want the venv to use your system OpenCV:

```bash
python - <<'PY'
import cv2
print(cv2.__version__)
for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line:
        print(line)
PY
```

Expected:

```text
4.6.0
GStreamer: YES (1.24.1)
```

If you see OpenCV 5.0.0 or GStreamer: NO, stop and remove pip OpenCV again.

3. Create the YOLO web server

Run this on Ubuntu:

```bash
mkdir -p src/object_tracking
cat > src/object_tracking/yolo_stream_server.py <<'PY'
import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from ultralytics import YOLO


DEFAULT_PIPELINE = (
    "udpsrc port=5600 buffer-size=1048576 ! "
    "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! "
    "rtpjitterbuffer latency=20 drop-on-latency=true ! "
    "rtph264depay ! h264parse ! avdec_h264 max-threads=2 ! videoconvert ! videoscale ! "
    "video/x-raw,width=640,height=360,format=BGR ! "
    "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
    "appsink sync=false drop=true max-buffers=1"
)


@dataclass
class SharedState:
    raw_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    jpeg_bytes: Optional[bytes] = None
    last_error: Optional[str] = None
    frame_count: int = 0
    yolo_count: int = 0
    fps: float = 0.0
    yolo_fps: float = 0.0
    lock: threading.Lock = threading.Lock()


state = SharedState()
app = FastAPI()


def opencv_gstreamer_enabled() -> bool:
    info = cv2.getBuildInformation()
    return "GStreamer:                   YES" in info or "GStreamer: YES" in info


def draw_fps(frame, fps, yolo_fps):
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


def camera_loop(pipeline: str, model_name: str, imgsz: int, conf: float, infer_every: int, jpeg_quality: int):
    global state

    print("OpenCV:", cv2.__version__)
    print("OpenCV GStreamer enabled:", opencv_gstreamer_enabled())
    print("Opening pipeline:")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        with state.lock:
            state.last_error = "Could not open GStreamer camera pipeline"
        print("ERROR: Could not open GStreamer camera pipeline")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO model:", model_name)
    print("YOLO device:", device)

    model = YOLO(model_name)
    model.to(device)

    last_annotated = None
    last_fps_time = time.time()
    last_yolo_time = time.time()
    frames_since_fps = 0
    yolo_since_fps = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            with state.lock:
                state.last_error = "No frame received from camera"
            time.sleep(0.01)
            continue

        frames_since_fps += 1

        # Run YOLO only every N frames for latency control.
        run_yolo = (state.frame_count % max(infer_every, 1) == 0)

        if run_yolo:
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=device,
            )

            annotated = frame.copy()

            if results and results[0].boxes is not None:
                names = results[0].names
                boxes = results[0].boxes

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    label = f"{names.get(cls_id, cls_id)} {score:.2f}"

                    x1, y1, x2, y2 = xyxy.tolist()
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            last_annotated = annotated
            yolo_since_fps += 1

        if last_annotated is None:
            display = frame.copy()
        else:
            # Reuse latest YOLO annotation between inference frames.
            display = last_annotated.copy()

        now = time.time()

        if now - last_fps_time >= 1.0:
            cam_fps = frames_since_fps / (now - last_fps_time)
            yolo_fps = yolo_since_fps / max(now - last_yolo_time, 1e-6)

            frames_since_fps = 0
            yolo_since_fps = 0
            last_fps_time = now
            last_yolo_time = now
        else:
            with state.lock:
                cam_fps = state.fps
                yolo_fps = state.yolo_fps

        draw_fps(display, cam_fps, yolo_fps)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ok_jpeg, jpeg = cv2.imencode(".jpg", display, encode_params)

        if ok_jpeg:
            with state.lock:
                state.raw_frame = frame
                state.annotated_frame = display
                state.jpeg_bytes = jpeg.tobytes()
                state.frame_count += 1
                state.yolo_count += 1 if run_yolo else 0
                state.fps = cam_fps
                state.yolo_fps = yolo_fps
                state.last_error = None


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>Unitree YOLO Stream</title>
        <style>
          body { font-family: Arial, sans-serif; background: #111; color: #eee; }
          img { max-width: 100%; height: auto; border: 2px solid #444; }
          .wrap { max-width: 1100px; margin: 24px auto; }
          code { background: #222; padding: 2px 5px; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <h1>Unitree G1 YOLO Stream</h1>
          <p>Annotated stream from Ubuntu vision server.</p>
          <img src="/stream.mjpg" />
          <p>
            Snapshot: <a href="/snapshot.jpg">/snapshot.jpg</a><br/>
            Health: <a href="/health">/health</a>
          </p>
        </div>
      </body>
    </html>
    """


@app.get("/health")
def health():
    with state.lock:
        return {
            "ok": state.jpeg_bytes is not None,
            "frame_count": state.frame_count,
            "yolo_count": state.yolo_count,
            "fps": state.fps,
            "yolo_fps": state.yolo_fps,
            "last_error": state.last_error,
            "opencv": cv2.__version__,
            "opencv_gstreamer": opencv_gstreamer_enabled(),
            "torch_cuda": torch.cuda.is_available(),
        }


@app.get("/snapshot.jpg")
def snapshot():
    with state.lock:
        data = state.jpeg_bytes

    if data is None:
        return Response(content=b"No frame yet", media_type="text/plain", status_code=503)

    return Response(content=data, media_type="image/jpeg")


def mjpeg_generator():
    last_sent_frame = -1

    while True:
        with state.lock:
            data = state.jpeg_bytes
            frame_count = state.frame_count

        if data is None or frame_count == last_sent_frame:
            time.sleep(0.001)
            continue

        last_sent_frame = frame_count

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" +
            data +
            b"\r\n"
        )


@app.get("/stream.mjpg")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    worker = threading.Thread(
        target=camera_loop,
        args=(args.pipeline, args.model, args.imgsz, args.conf, args.infer_every, args.jpeg_quality),
        daemon=True,
    )
    worker.start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
PY
```

4. Run the YOLO server on Ubuntu

Run this on Ubuntu:

```bash
cd ~/Documents/project
uv run g1 local start \
  --model yolov8n.pt \
  --imgsz 320 \
  --conf 0.35 \
  --infer-every 1 \
  --jpeg-quality 60 \
  --host 0.0.0.0 \
  --port 8000
```

Then on your MacBook, open:

```text
http://192.168.0.122:8000
```

You should see the Unitree camera video with YOLO boxes.

Why this is better than Streamlit

This design keeps the server loop simple:

Read latest frame
Run YOLO sometimes
Draw boxes
Encode JPEG
Serve latest JPEG repeatedly

It avoids Streamlit rerun overhead. It also lets your MacBook view the output without X11, without SSH GUI forwarding, and without decoding the robot stream itself.

Latency knobs

Start with these:

```bash
--model yolov8n.pt
--imgsz 320
--infer-every 1
--jpeg-quality 60
```

If it is still laggy, make it faster:

```bash
--imgsz 320
--infer-every 3
--jpeg-quality 55
```

If boxes are too inaccurate, make it better:

```bash
--imgsz 640
--infer-every 1
--jpeg-quality 75
```

For now, I would run:

```bash
--imgsz 320 --infer-every 1 --jpeg-quality 60
```

That is a good compromise.

Critical design rule

Do not block the camera loop waiting for the browser.

The browser should always receive the latest annotated frame. The robot/tracking system should always drop stale frames.

That is why the GStreamer pipeline uses:

```text
rtpjitterbuffer latency=20 drop-on-latency=true
queue max-size-buffers=1 leaky=downstream after video/x-raw
appsink sync=false drop=true max-buffers=1
```

And the server keeps only:

```python
state.jpeg_bytes = latest annotated JPEG
```

No backlog.

What this gives you

This is now your VLM substitute layer:

YOLOv8 detects objects
OpenCV draws boxes
FastAPI relays annotated stream
MacBook watches from browser

Later, replace or augment YOLO with:

fine-tuned VLM
custom object detector
tracker
intercept planner
robot control loop

But for now this gets you the core visual loop working efficiently on a headless Ubuntu server.

Troubleshooting: if `/snapshot.jpg` returns `503` and the server log never prints `Loading YOLO model`, the camera loop has not produced a decoded frame yet. Check the GStreamer pipeline and make sure nothing drops RTP/H264 packets before `rtph264depay`.
