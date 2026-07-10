# Humanoid Robot Grasping

The RealSense-guided right-arm pregrasp pipeline is implemented as a disarmed-by-default robot depth/arm service plus a GB10 hardware-depth fusion and IK runtime. See [docs/ARM_TRACKING_RUNBOOK.md](docs/ARM_TRACKING_RUNBOOK.md) for installation, calibration, dry-run, REST schemas, port forwarding, and operator-gated hardware stages.

The GB10 startup output prints the exact browser URL and SSH tunnel command for a MacBook. Its research console combines live RGB/depth views with pipeline timing, detections, 3D targets, IK/arm state, rejection diagnostics, and downloadable JSONL telemetry stored under `runs/research/arm_tracking/`.

For the shortest, discoverable command interface, use `uv run g1`. It groups setup, robot services, arm commissioning, streams, tuning, data, and training while preserving the existing host-specific scripts for automation. Start with `uv run g1 --help` and see [docs/COMMANDS.md](docs/COMMANDS.md).

The three runtime roles and their network ownership are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Joint-order validation and movement-free tuning helpers are documented in [docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md). Use `uv run g1-tune joint-audit`, `uv run g1-tune analyze runs/research/arm_tracking`, and `uv run g1-tune compare runs/research/arm_tracking` to verify the 29-DOF contract and compare controlled dry-run trials.

Before tuning moving-object tracking, use the separate camera-free right-arm commissioning workflow in [docs/ARM_COMMISSIONING_RUNBOOK.md](docs/ARM_COMMISSIONING_RUNBOOK.md). The robot-local wizard and `g1-arm` CLI provide verified 0.01 rad one-joint jogs, 0.05 rad operator-approved stages, a 0.30 rad per-joint session envelope, deadman release, event logs, and explicit promotion of a measured home pose.

## Setup

From the project root:

```bash
git switch aarav
uv run g1 setup gb10
```

The GB10 setup installs the vision, training, calibration, and arm-IK groups in Python 3.12. The robot uses the isolated `robot/` environment for depth and arm services while the compressed GStreamer relay remains dependency-free. Install locomotion separately on other robot-network hosts when needed:

```bash
uv sync --only-group loco --locked
```

If loco setup fails while building `cyclonedds`, install/configure the CycloneDDS system library on the robot-network host first, then rerun the loco sync. The Python package needs the C library visible through `CYCLONEDDS_HOME` or `CMAKE_PREFIX_PATH`.

Loco commands are run from the server that is connected to the robot network, not necessarily on the robot itself. Pass the robot LAN IP to `uv run loco`. The Unitree SDK is pinned to an official Git revision in the lockfile; the robot SSH target is separate from that Python dependency.

```bash
uv sync --only-group loco --locked
uv run loco <robot_lan_ip> stop_move
```

## Vision runbook

If you need the SDK2 visual frame path, install the loco group too and run this from the machine on the robot network. This does not move the robot:

```bash
uv run vision <g1_ip> --sdk2-video --sdk2-timeout 3
```

If it works, this writes `runs/vision_test/snapshot.jpg` and `runs/vision_test/camera.json` using `videohub.GetImageSample`. If it fails with `cyclonedds`, install/configure CycloneDDS on that host. If it returns an SDK2 videohub error, the robot may expose camera through onboard RealSense/ROS or another Unitree package rather than SDK2 videohub.

1) Make sure you know the robot LAN IP.
2) Use these commands from the project root on the machine that is connected to the robot network:

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
ssh <robot_ssh_user>@<robot_ssh_host> "cd /home/<robot_ssh_user>/project && uv run g1 setup local && uv run vision <g1_ip> --no-ssh"
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
uv run python scripts/local/manual.py --camera 0 --output runs/object_manual_test
```

## Vision FastAPI stream

The camera stays owned by Unitree's `videohub_pc4` task. Video remains compressed until it reaches the GB10:

```text
videohub_pc4 -> RTP multicast 230.1.1.1:1720 -> robot relay -> GB10 UDP 5600
             -> GStreamer decode -> fine-tuned YOLOv8 -> FastAPI/viewer -> Mac SSH tunnel
```

### Robot: compressed RTP relay

Do not open `/dev/video4`; `videohub_pc4` already owns it. Relay its multicast RTP packets to the GB10 address:

```bash
uv run g1 robot start --client-ip 192.168.0.66 --token-file <TOKEN>
```

The default source is multicast `230.1.1.1:1720` on `wlan0`; the default destination port is UDP `5600`. Override `ROBOT_INTERFACE`, `MULTICAST_GROUP`, `MULTICAST_PORT`, or `CLIENT_PORT` only when the robot network differs.

The relay does not decode, resample, or rewrite frame timing. Camera mode remains owned by `videohub_pc4`; the GB10 `/health` endpoint verifies that the received stream meets the expected 30 FPS target.

### GB10: verify relay

Run the one-time GB10 setup:

```bash
uv run g1 setup gb10
```

Optionally confirm that the GB10 can decode the live stream to a headless sink. This writes no frames to disk:

```bash
uv run g1 gb10 relay-test
```

Then start inference and the website:

```bash
uv run g1 gb10 start
```

The default checkpoint is the fine-tuned plush-animal model:

```text
models/plushie_detector/yolov8n_plushie_mvp/weights/best.pt
```

The default tracking profile preserves the robot feed at `1280x720`, processes every frame, and serves the annotated stream at 30 FPS with JPEG quality 75. Override it only when bandwidth or inference load requires it:

```bash
VISION_WIDTH=640 VISION_HEIGHT=360 VISION_FPS=30 JPEG_QUALITY=60 uv run g1 gb10 start
```

The GB10 uses external `gst-launch-1.0` for decoding, so pip OpenCV does not need GStreamer support. Useful endpoints are:

```text
http://GB10:8000/stream.mjpg
http://GB10:8000/snapshot.jpg
http://GB10:8000/detections
http://GB10:8000/tracks
http://GB10:8000/health
http://GB10:8080/unitree_dual_viewer.html?single=1
```

`/health` reports `inference_status` as `warming_up`, `ready`, or `error`, plus measured `fps`, `yolo_fps`, and `fps_target_met`. The 30 FPS health target allows normal timing jitter down to 27 FPS. Wait for inference `ready` and `fps_target_met: true` before evaluating detections; startup performs an explicit CUDA warmup and records any inference-thread exception instead of failing silently.

### Laptop: SSH tunnel and browser

From the Mac, forward the viewer and processed inference stream:

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -L 8000:127.0.0.1:8000 \
  USER@GB10_HOST
```

Then open:

```text
http://127.0.0.1:8080/unitree_dual_viewer.html?single=1
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
models/plushie_detector/yolov8n_plushie_mvp/weights/best.pt
models/plushie_detector/yolo11x_plushie/weights/best.pt
models/plushie_detector/yolo11x_plushie/weights/last.pt
```

Capture frames from the running stream server:

```bash
NOTE="plushie moving left to right" COUNT=20 INTERVAL=0.25 \
  uv run g1 data capture
```

This calls `/capture` and writes raw images plus metadata to:

```text
runs/captures/plushie/YYYYMMDD_HHMMSS/images/frame_000001.jpg
runs/captures/plushie/YYYYMMDD_HHMMSS/metadata.jsonl
```

Label the captured frames in CVAT, Roboflow, or Label Studio, export YOLO detection labels, then place images/labels under `data/plushie`.

Install public bootstrap datasets in the background:

```bash
nohup uv run g1 data install > runs/dataset_install/nohup.log 2>&1 &
tail -f runs/dataset_install/install_all_plushie_datasets.log
```

The installer currently pulls COCO 2017 `teddy bear` plus Open Images `Teddy bear`, maps positives to `plushie`, and adds 1:1 hard negatives from confusing toy/animal/soft-object classes. Roboflow sources are listed but skipped until `ROBOFLOW_API_KEY` is provided and each dataset license is acceptable.

Create train-only augmented images that mimic the G1 feed: blur, motion smear, JPEG compression, sensor noise, gray-floor 1280x720 canvases, and smaller object scale. Validation data is left untouched.

```bash
uv run g1 data augment
```

After the default public-dataset install and augmentation pass, the local YOLO set is currently:

```text
train images/labels: 25288
val images/labels:     225
augmented train rows: 18966
```

Useful capped smoke test:

```bash
uv run g1 data augment --max-source-images 20 --aug-per-image 3
```

Train the plushie detector:

```bash
uv run g1 train detector
```

Fast MVP training pass for same-day testing:

```bash
MODEL=yolov8n.pt NAME=yolov8n_plushie_mvp EPOCHS=25 IMGSZ=960 BATCH=128 WORKERS=10 CACHE=disk RAM_RESERVE_GB=24 CPU_RESERVE_PERCENT=50 SAVE_PERIOD=5 PATIENCE=6 uv run g1 train guarded
```

Useful overrides:

```bash
MODEL=yolo11x.pt EPOCHS=160 IMGSZ=1280 BATCH=16 SAVE_PERIOD=5 uv run g1 train detector
```

Recommended GB10 server training pass for the current COCO-derived dataset:

```bash
MODEL=yolo11x.pt EPOCHS=160 IMGSZ=1280 BATCH=16 WORKERS=10 CACHE=auto RAM_RESERVE_GB=16 CPU_RESERVE_PERCENT=50 SAVE_PERIOD=5 PATIENCE=40 uv run g1 train guarded
```

For this augmented dataset, avoid `BATCH=-1 CACHE=ram` at `IMGSZ=1280`. AutoBatch probes oversized batches, and decoded RAM cache at 1280 can exceed the 120 GiB usable memory budget before model/data-loader overhead. `CACHE=auto` uses RAM cache only when the estimate fits the configured reserve; otherwise it uses disk cache and lets the OS page cache consume spare RAM safely. The guarded launcher adds cgroup limits: memory is capped to total RAM minus `RAM_RESERVE_GB`, swap is disabled for the training unit, and CPU quota leaves `CPU_RESERVE_PERCENT` for the rest of the system. On the GB10, `BATCH=16` was the best observed balance: it used about 58.5 GiB GPU memory without pushing host RAM into the danger zone. `BATCH=24` used about 87.6 GiB GPU memory but drove host memory low enough to touch swap, so do not use it for this dataset.

After adding Unitree-camera frames, fine-tune from the latest checkpoint for another 80-120 epochs:

```bash
MODEL=models/plushie_detector/yolo11x_plushie/weights/best.pt EPOCHS=120 IMGSZ=1280 BATCH=16 WORKERS=10 CACHE=auto RAM_RESERVE_GB=16 CPU_RESERVE_PERCENT=50 NAME=yolo11x_plushie_unitree uv run g1 train guarded
```

Resume the latest interrupted run:

```bash
RESUME=1 uv run g1 train detector
```

Evaluate the trained detector:

```bash
uv run g1 train evaluate
```

Run the plushie stream. This uses `models/plushie_detector/yolo11x_plushie/weights/best.pt` if it exists. If no trained model exists yet, put a fallback model under `models/pretrained/` or set `MODEL` to an explicit path under `models/`.

```bash
uv run g1 gb10 start
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
uv run loco 192.168.0.213 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.213 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco --diagnose 192.168.0.4
```

For G1 loco, `--loco-service-name auto` is the default and patches the SDK to use `ai_sport`. Use `--loco-service-name sport` only when explicitly testing older firmware/service behavior.
For DDS init, `--dds-config-mode no_trace` is the default. The Unitree SDK's original `unitree` config mode is kept for debugging, but it has been observed to crash with `SIGABRT` during `ChannelFactoryInitialize` on the G1 test machine.

By default, loco resolves the local DDS interface with `ip route get <robot_ip>`. If DDS picks the wrong NIC, pass it explicitly:

```bash
uv run loco 192.168.0.4 stop_move --interface eno1
uv run loco 192.168.0.4 stop_move --network-interface eno1
```

`--interface auto` is the default. Use `uv run loco --diagnose <robot_ip>` to print the resolved interface, Python paths, Unitree SDK path/version, CycloneDDS path/version, and relevant DDS environment variables.

If `LocoClient` fails during DDS topic creation, probe the specific DDS topics without sending robot commands:

```bash
uv run loco --dds-probe 192.168.0.4
uv run loco --dds-probe 192.168.0.4 --interface eno1 --loco-service-name ai_sport
```

### Unitree DDS troubleshooting

If `LocoClient` fails while creating a CycloneDDS topic, first run the project-free minimal constructor test:

```bash
uv run python scripts/robot/loco_minimal.py --robot-ip 192.168.0.4 --loco-service-name ai_sport
uv run python scripts/robot/loco_minimal.py --interface eno1 --loco-service-name ai_sport
```

You can also run the same smoke test through the main CLI without sending movement commands:

```bash
uv run loco --smoke-loco 192.168.0.4 --interface eno1 --loco-service-name ai_sport
uv run loco --smoke-loco-subprocess 192.168.0.4 --interface eno1 --loco-service-name ai_sport
uv run loco --smoke-loco-config-sweep 192.168.0.4 --interface eno1 --loco-service-name ai_sport
```

Prefer `--smoke-loco-subprocess` when debugging native crashes such as `*** buffer overflow detected ***`; it reports whether the child process exited normally or was killed by a native signal.
Use `--smoke-loco-config-sweep` when the subprocess dies during `ChannelFactoryInitialize`. It tests `unitree`, `no_trace`, `simple`, and `autodetermine` DDS config modes in separate child processes.

The Unitree SDK config can try to write CycloneDDS tracing to `/tmp/cdds.LOG`. This CLI patches that path to `/tmp/unitree_cdds_<uid>_<pid>.log` before DDS initialization. To choose a specific writable file:

```bash
uv run loco --smoke-loco-subprocess 192.168.0.4 --interface eno1 --loco-service-name ai_sport --cyclonedds-log-file /tmp/ucdds.log
```

If one config mode passes, use it for later commands:

```bash
uv run loco 192.168.0.213 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
```

If every config mode exits with `SIGABRT` before `Constructing G1 LocoClient`, the failure is in CycloneDDS domain initialization, not in the repo movement wrapper or G1 service name patch. Rebuild or reinstall the native CycloneDDS and `unitree_sdk2py` stack before trying robot movement again.

If the minimal script fails, the problem is below this project: fix the Unitree SDK checkout, CycloneDDS/Python environment, DDS domain/interface, or robot firmware/SDK compatibility. Inspect `sys.path`, remove duplicate SDK installs, and reinstall exactly one `unitree_sdk2py` source cleanly.

If the minimal script passes but `uv run loco ...` fails, look for mixed-client imports or fallback construction in this repo. The default backend is `g1_loco`; `go2_sport` is only constructed when explicitly requested:

```bash
uv run loco 192.168.0.4 stop_move --backend g1_loco
uv run loco 192.168.0.4 --backend g1_loco_minimal
```

During a normal `stop_move`, stderr should contain exactly one Unitree client construction line: `Constructing G1 LocoClient`.

If `stop_move` reaches `[ClientStub] send request error`, DDS and `LocoClient` construction have already succeeded. Use the read-only probe before sending more movement commands:

```bash
uv run loco 192.168.0.213 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.213 probe_loco --interface enP7s7 --loco-service-name sport --dds-config-mode no_trace
```

If both probes return `3102` (`Request sending error`) on read-only methods, put the robot into high-level sport/ai-sport mode with the controller and retry. At that point the failure is the robot RPC server not responding on `rt/api/<service>/request`, not DDS initialization. The next useful checks are whether the robot firmware exposes the high-level loco RPC service at all and whether motion mode is enabled on the robot side.

If MotionSwitcher also returns `3102`, first classify the server-to-robot path from the machine that will send commands:

```bash
PYTHONPATH=$PWD/src python3 scripts/dev/robot_network.py \
  --interfaces enP7s7,wlan0 \
  --ping-ip 192.168.0.213 \
  --domain-id 0 \
  --timeout 15.0
```

The probe script sets `PYTHONPATH=./src` for child commands, checks `object_tracking` imports before DDS, and reports per-interface classifications. `--ping-ip` is only a network diagnostic; Unitree DDS discovery uses `--interfaces`, `--domain-id`, and the SDK service names.

Direct server-side read-only checks:

```bash
uv run loco \
  --interface enP7s7 \
  --domain-id 0 \
  --timeout 15.0 \
  --dds-config-mode no_trace \
  diagnose \
  --loco-service-name sport

uv run loco \
  --interface wlan0 \
  --domain-id 0 \
  --timeout 15.0 \
  --dds-config-mode no_trace \
  probe_loco \
  --loco-service-name sport
```

Diagnosis rules:
- Import check fails: repo packaging/import issue; DDS was not tested.
- One interface returns SDK RPC code `0`: use that interface for Unitree DDS.
- `3102`: request sending/network/interface issue.
- `3103`: API not registered or wrong service name; try `sport` and `ai_sport`.
- `3104`: timeout/discovery or service unavailable.
- Server-side fails but robot-local succeeds: do not send Unitree DDS from the server; run a robot-side motion daemon and send HTTP/WebSocket commands over Wi-Fi.

### Server-to-robot command path

If `uv run g1 inspect dds` shows no DDS discovery output on the server interfaces, direct server-side Unitree SDK2 DDS is not reaching the robot. In that state, `probe_loco`, `check_motion_mode`, and `stop_move` from the server will return `3102` because no robot RPC participant is discovered.

Use the robot-side HTTP bridge instead. It keeps DDS local to the robot and lets the external server send ordinary HTTP JSON over Wi-Fi.

On the robot:

```bash
cd ~/humanoid-robot-grasping
git pull
uv run g1 robot command serve
```

From the server:

```bash
uv run g1 robot command health
uv run g1 robot command mode
uv run g1 robot command probe
uv run g1 robot command stop
```

If `wlan0` does not work inside the bridge, restart it with:

```bash
uv run g1 robot command serve --interface eth0
```

The server still talks to `192.168.0.213:8765`; only the robot-local DDS interface changes.

Movement is enabled by default for this robot-local bridge. Use `--read-only` when starting the bridge if you want to disable movement endpoints.

```bash
uv run g1 robot command smoke-move
```

Small bounded arm test:

```bash
uv run g1 robot command arms-up 0.15
uv run g1 robot command forward 0.05 --duration 0.4 --ramp 0.15
uv run g1 robot command move --vx 0.05 --vy 0 --omega 0 --duration 0.4 --ramp 0.15
uv run g1 robot command stop
```

Avoid interactive keyboard control for now. Use one bounded command at a time
(`forward`, `move`, or `arms-up`), then send `stop` before the next test.

`amount` is the fraction of the forward arm target. Start around `0.1` to `0.2`. Arm and velocity motion use smoothstep easing so they ease in/out instead of snapping to a linear ramp.

Only use robot-local checks to isolate low-level robot networking after server-side tests are exhausted:

```bash
ssh unitree@192.168.0.213
cd ~/humanoid-robot-grasping
git pull
python3 scripts/dev/robot_network.py --interfaces eth0,wlan0 --ping-ip 192.168.123.1
```

If the robot does not show an obvious `ai_sport`/`loco` Linux service to start manually, use the SDK motion switcher path:

```bash
uv run loco 192.168.0.213 check_motion_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.213 select_ai_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.213 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
```

`check_motion_mode` is read-only. `select_ai_mode` calls `MotionSwitcherClient.SelectMode("ai")`, so only run it when the robot is physically safe and the controller/e-stop is ready.

### Robot-local shoulder pitch arm test

This script runs on the robot itself and uses low-level SDK2 topics. It releases high-level motion mode, reads the current full-body posture from `rt/lowstate`, holds every motor at that posture, and only offsets the selected shoulder pitch joint(s).

On the robot:

```bash
cd ~/humanoid-robot-grasping
git pull

uv run g1 robot command shoulder-pitch \
  --side both \
  --sign 1 \
  --delta 0.25 \
  --ramp-seconds 3.0 \
  --hold-seconds 5.0
```

Dry-run without DDS or movement:

```bash
uv run g1 robot command shoulder-pitch --smoke --side both --sign 1 --delta 0.25
```

If the shoulder pitch direction is backwards, retry with `--sign -1` or `--direction negative`. If left/right need opposite directions, use `--left-sign 1 --right-sign -1` or the reverse. Start with a smaller `--delta 0.1` if you only want a small motion check.

After a read-only probe returns `ok: true`, a tiny movement smoke test is available but guarded:

```bash
uv run loco 192.168.0.213 smoke_move \
  --interface enP7s7 \
  --loco-service-name sport \
  --dds-config-mode no_trace
```
