# Humanoid Robot Grasping

## Setup

From the project root:

```bash
git switch aarav
uv sync
```

This installs vision dependencies only by default (`opencv-python`). Loco extras are optional:

```bash
uv sync --extra loco
```

Loco uses `unitree-sdk2==1.0.1` (package path: `unitree_sdk2`).

## Vision runbook

First try the SDK2 visual frame path from the machine on the robot network. This does not move the robot:

```bash
uv run vision <g1_ip> --sdk2-video --sdk2-timeout 3
```

If it works, this writes `runs/vision_test/snapshot.jpg` and `runs/vision_test/camera.json` using `videohub.GetImageSample`. If it fails with `cyclonedds`, install/configure CycloneDDS on that host. If it returns an SDK2 videohub error, the robot may expose camera through onboard RealSense/ROS or another Unitree package rather than SDK2 videohub.

1) Make sure you know the robot LAN IP.
2) Use these commands from `/home/aarav/Documents/project`:

Headless snapshot (fastest smoke test, no display needed):

```bash
uv run vision 192.168.0.4
```

SSH-friendly form (vision can parse target args directly):

```bash
uv run vision 192.168.0.4 unitree@192.168.0.100
uv run vision 192.168.0.4 unitree 192.168.0.100
```

Important for G1: these SSH args are for RTSP tunneling fallbacks. The default G1 camera is UDP/GStreamer, so they won’t change transport.
If SSH is needed, run the vision command on the SSH host itself and use `--no-ssh`.

If reachable, this opens one frame and saves:
- `runs/vision_test/snapshot.jpg`
- `runs/vision_test/camera.json`

### Run from a separate machine (SSH host)

If this repo is not on the same host as the robot, copy it over first:

```bash
rsync -av \
  --exclude .venv \
  --exclude .git \
  /home/aarav/Documents/project/ \
  <robot_ssh_user>@<robot_ssh_host>:/home/<robot_ssh_user>/project/
```

Then run from the remote host (required for G1 UDP/GStreamer):

```bash
ssh <robot_ssh_user>@<robot_ssh_host> "cd /home/<robot_ssh_user>/project && uv sync --extra loco && uv run vision <g1_ip> --no-ssh"
```

Optional: track headless for testing:

```bash
ssh <robot_ssh_user>@<robot_ssh_host> \
  "cd /home/<robot_ssh_user>/project && uv run vision <g1_ip> --no-ssh --bbox \"320 180 200 200\" --max-frames 300"
```

Run tracker (headless, scripted target box):

```bash
uv run vision 192.168.0.4 --bbox "320 180 200 200" --max-frames 300
```

If you are on a machine without GUI, `--bbox` is required.  
If you have a display, you can omit `--bbox` and add `--show` for manual ROI selection.

Debug first if it fails:

```bash
uv run vision 192.168.0.4 --list-cameras
uv run vision 192.168.0.4 --diagnose
```

Commonly useful options:

```bash
uv run vision 192.168.0.4 --camera-url "udpsrc port=5600 ! application/x-rtp,media=(string)video ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
uv run vision 192.168.0.4 --no-ssh
uv run vision 192.168.0.4 --show
```

Notes:
- The current vision fallback still includes GStreamer/UDP candidates, but port 5600 is not confirmed as the G1 camera default.
- That UDP pipeline listens locally on port 5600 (`udpsrc port=5600`). Do not bind `udpsrc address=` to the robot IP; only set `G1_CAMERA_BIND_ADDRESS` when you need to bind a specific local interface address.
- `opencv-python` wheels commonly lack GStreamer support. `uv run vision <robot_ip> --diagnose` reports `opencv.gstreamer`; if it is false, run on a host with a GStreamer-enabled OpenCV build or pass an RTSP/file source explicitly.
- `--no-ssh` disables SSH tunnel probing (safe default for GStreamer/UDP).
- If using RTSP URLs manually, use `--ssh-host`/`--ssh-user` if port forwarding is needed.
- `--diagnose` writes probe JSON including network + camera candidate status.

Example local webcam path (non-Robot testing):

```bash
uv run python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
```

## YOLO FastAPI stream

For low-latency browser preview, use the headless FastAPI MJPEG server instead of Streamlit. The Ubuntu server receives the Unitree H264 RTP stream on UDP port `5600`, decodes it with GStreamer/OpenCV, runs YOLO, draws boxes, and serves the latest annotated JPEG at `/stream.mjpg`.

On the Ubuntu vision box, run this once:

```bash
cd ~/Documents/project
./scripts/setup_vision_server.sh
```

Then start the stream server:

```bash
./scripts/run_yolo_stream.sh
```

The defaults are already set for the current plan:

```text
model=yolov8n.pt
imgsz=320
conf=0.35
infer_every=1
jpeg_quality=60
max_det=20
opencv_threads=16
torch_threads=16
stream_fps=0 (unbounded; sends each new JPEG immediately)
host=0.0.0.0
port=8000
```

Then open this from the MacBook:

```text
http://192.168.0.122:8000
```

Useful endpoints:

```text
/stream.mjpg
/snapshot.jpg
/capture
/detections
/tracks
/health
```

If `/snapshot.jpg` returns `503` and the server log never prints `Loading YOLO model`, the camera loop has not produced a decoded frame yet. Check the GStreamer pipeline and make sure nothing drops RTP/H264 packets before `rtph264depay`.

The setup script uses `uv venv --system-site-packages .venv`, so the project keeps the standard `.venv` name while still seeing Ubuntu's system OpenCV with GStreamer enabled. It also removes pip OpenCV wheels because those usually do not include GStreamer.

To trade a little detector update rate for more camera/browser FPS without editing files:

```bash
INFER_EVERY=2 JPEG_QUALITY=55 ./scripts/run_yolo_stream.sh
```

Keep the Unitree relay running separately so compressed video reaches the Ubuntu box:

```bash
gst-launch-1.0 -v \
  udpsrc multicast-group=230.1.1.1 address=0.0.0.0 port=1720 auto-multicast=true multicast-iface=wlan0 buffer-size=1048576 ! \
  "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000" ! \
  queue ! \
  udpsink host=192.168.0.122 port=5600 sync=false async=false
```

## Plushie detector pipeline

The runtime perception loop should stay detector + tracker only. The custom YOLO detector finds plushie boxes, the tracker assigns `track_id` and estimates pixel velocity, and later language/LLM code should consume structured JSON from `/detections` and `/tracks` rather than raw frames. A VLM can still be useful for dataset labeling or rare semantic fallback, but it is not part of the real-time loop.

Dataset layout:

```text
data/plushie/
  plushie.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Training artifacts are intentionally rooted under `models/`:

```text
models/plushie_detector/yolov8n_plushie/weights/best.pt
```

Capture frames from the running stream server:

```bash
NOTE="plushie moving left to right" COUNT=20 INTERVAL=0.25 \
  ./scripts/capture_training_frames.sh
```

This calls `/capture` and writes raw images plus metadata to:

```text
runs/captures/plushie/YYYYMMDD_HHMMSS/images/frame_000001.jpg
runs/captures/plushie/YYYYMMDD_HHMMSS/metadata.jsonl
```

Label the captured frames in CVAT, Roboflow, or Label Studio, export YOLO detection labels, then place images/labels under `data/plushie`.

Train the plushie detector:

```bash
./scripts/train_plushie_detector.sh
```

Useful overrides:

```bash
MODEL=yolov8n.pt EPOCHS=80 IMGSZ=640 BATCH=16 ./scripts/train_plushie_detector.sh
```

Evaluate the trained detector:

```bash
./scripts/eval_plushie_detector.sh
```

Run the plushie stream. This uses `models/plushie_detector/yolov8n_plushie/weights/best.pt` if it exists and falls back to `yolov8n.pt` otherwise:

```bash
./scripts/run_plushie_stream.sh
```

Check structured perception output:

```bash
curl http://127.0.0.1:8000/detections
curl http://127.0.0.1:8000/tracks
```

The current instruction parser stub is rule-based and does not call an LLM:

```python
from object_tracking.instruction_parser import parse_instruction

parse_instruction("stop the stuffed animal")
```

## Unitree SDK2 loco

Find the robot current LAN IP:

```bash
uv run g1-scan
```

If the router blocks ping or the robot is quiet, run a slower scan:

```bash
uv run g1-scan --include-sleeping --connect-timeout 0.5
```

Run commands with robot IP:

```bash
uv run loco 192.168.0.4 stand_up
uv run loco 192.168.0.4 balance_stand
uv run loco 192.168.0.4 move --velocity "0.2 0 0 1.0"
uv run loco 192.168.0.4 stop_move
uv run loco 192.168.0.4 --diagnose
```

`--network-interface` is only needed if route detection fails.
