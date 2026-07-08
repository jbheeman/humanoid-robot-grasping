# Humanoid Robot Grasping

## Setup

From the project root:

```bash
git switch aarav
uv sync
```

This installs the default runtime plus the Unitree loco dependency group. The Unitree SDK checkout must exist next to this repo at `../unitree_sdk2_python`; uv installs it editable as `unitree-sdk2py==1.0.1` (import path: `unitree_sdk2py`). If your SDK checkout is somewhere else, update the `unitree-sdk2py` path in `pyproject.toml` before running `uv sync`.

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
ssh <robot_ssh_user>@<robot_ssh_host> "cd /home/<robot_ssh_user>/project && uv sync && uv run vision <g1_ip> --no-ssh"
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

For low-latency browser preview, use the headless FastAPI MJPEG server instead of Streamlit. By default it reads the direct RealSense/V4L2 color stream at `640x480@30`, scales to `640x360`, runs YOLO, draws boxes/tracks, and serves the latest annotated JPEG at `/stream.mjpg`.

On the Ubuntu vision box, run this once:

```bash
cd ~/Documents/project
./scripts/setup_vision_server.sh
```

Then start the 30 FPS direct-camera stream server:

```bash
./scripts/run_yolo_stream.sh
```

The defaults are already set for the current plan:

```text
pipeline=v4l2src device=/dev/video0 ... framerate=30/1
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

If `/dev/video0` is not the RealSense RGB device, find the correct node and override it:

```bash
v4l2-ctl --list-devices
DEVICE=/dev/videoX ./scripts/run_realsense_stream.sh
```

The old Unitree multimedia UDP path is still available, but it is expected to be capped around 15 FPS at this resolution:

```bash
./scripts/run_unitree_udp_stream.sh
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

Training artifacts and pretrained model downloads are intentionally rooted under `models/`:

```text
models/pretrained/yolo11x.pt
models/plushie_detector/yolo11x_plushie/weights/best.pt
models/plushie_detector/yolo11x_plushie/weights/last.pt
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

Install public bootstrap datasets in the background:

```bash
nohup ./scripts/install_all_plushie_datasets.sh > runs/dataset_install/nohup.log 2>&1 &
tail -f runs/dataset_install/install_all_plushie_datasets.log
```

The installer currently pulls COCO 2017 `teddy bear` plus Open Images `Teddy bear`, maps positives to `plushie`, and adds 1:1 hard negatives from confusing toy/animal/soft-object classes. Roboflow sources are listed but skipped until `ROBOFLOW_API_KEY` is provided and each dataset license is acceptable.

Create train-only augmented images that mimic the G1 feed: blur, motion smear, JPEG compression, sensor noise, gray-floor 1280x720 canvases, and smaller object scale. Validation data is left untouched.

```bash
./scripts/augment_plushie_dataset.sh
```

After the default public-dataset install and augmentation pass, the local YOLO set is currently:

```text
train images/labels: 25288
val images/labels:     225
augmented train rows: 18966
```

Useful capped smoke test:

```bash
./scripts/augment_plushie_dataset.sh --max-source-images 20 --aug-per-image 3
```

Train the plushie detector:

```bash
./scripts/train_plushie_detector.sh
```

Fast MVP training pass for same-day testing:

```bash
MODEL=yolov8n.pt NAME=yolov8n_plushie_mvp EPOCHS=25 IMGSZ=960 BATCH=128 WORKERS=10 CACHE=disk RAM_RESERVE_GB=24 CPU_RESERVE_PERCENT=50 SAVE_PERIOD=5 PATIENCE=6 ./scripts/run_guarded_plushie_training.sh
```

Useful overrides:

```bash
MODEL=yolo11x.pt EPOCHS=160 IMGSZ=1280 BATCH=16 SAVE_PERIOD=5 ./scripts/train_plushie_detector.sh
```

Recommended GB10 server training pass for the current COCO-derived dataset:

```bash
MODEL=yolo11x.pt EPOCHS=160 IMGSZ=1280 BATCH=16 WORKERS=10 CACHE=auto RAM_RESERVE_GB=16 CPU_RESERVE_PERCENT=50 SAVE_PERIOD=5 PATIENCE=40 ./scripts/run_guarded_plushie_training.sh
```

For this augmented dataset, avoid `BATCH=-1 CACHE=ram` at `IMGSZ=1280`. AutoBatch probes oversized batches, and decoded RAM cache at 1280 can exceed the 120 GiB usable memory budget before model/data-loader overhead. `CACHE=auto` uses RAM cache only when the estimate fits the configured reserve; otherwise it uses disk cache and lets the OS page cache consume spare RAM safely. The guarded launcher adds cgroup limits: memory is capped to total RAM minus `RAM_RESERVE_GB`, swap is disabled for the training unit, and CPU quota leaves `CPU_RESERVE_PERCENT` for the rest of the system. On the GB10, `BATCH=16` was the best observed balance: it used about 58.5 GiB GPU memory without pushing host RAM into the danger zone. `BATCH=24` used about 87.6 GiB GPU memory but drove host memory low enough to touch swap, so do not use it for this dataset.

After adding Unitree-camera frames, fine-tune from the latest checkpoint for another 80-120 epochs:

```bash
MODEL=models/plushie_detector/yolo11x_plushie/weights/best.pt EPOCHS=120 IMGSZ=1280 BATCH=16 WORKERS=10 CACHE=auto RAM_RESERVE_GB=16 CPU_RESERVE_PERCENT=50 NAME=yolo11x_plushie_unitree ./scripts/run_guarded_plushie_training.sh
```

Resume the latest interrupted run:

```bash
RESUME=1 ./scripts/train_plushie_detector.sh
```

Evaluate the trained detector:

```bash
./scripts/eval_plushie_detector.sh
```

Run the plushie stream. This uses `models/plushie_detector/yolo11x_plushie/weights/best.pt` if it exists. If no trained model exists yet, put a fallback model under `models/pretrained/` or set `MODEL` to an explicit path under `models/`.

```bash
./scripts/run_plushie_stream.sh
```

Known dataset sources are tracked in [docs/PLUSHIE_DATASETS.md](docs/PLUSHIE_DATASETS.md).

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
