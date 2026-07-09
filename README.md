# Humanoid Robot Grasping

## Setup

From the project root:

```bash
git switch aarav
./scripts/setup_vision_server.sh
```

For the headless camera/FastAPI vision server on the G1, prefer `./scripts/setup_vision_server.sh`; it installs the `vision` group without CycloneDDS, Unitree SDK2 Python, Torch, Ultralytics, or CUDA packages. Install the other groups only where needed:

```bash
./scripts/setup_vision_server.sh
uv sync --only-group train --locked
./scripts/ensure_unitree_sdk_path.sh && uv sync --only-group loco --locked
```

If loco setup fails while building `cyclonedds`, install/configure the CycloneDDS system library on the robot-network host first, then rerun the loco sync. The Python package needs the C library visible through `CYCLONEDDS_HOME` or `CMAKE_PREFIX_PATH`.

Loco commands are run from the server that is connected to the robot network, not necessarily on the robot itself. Pass the robot LAN IP to `uv run loco`. The local editable `unitree-sdk2py==1.0.1` checkout must exist either at `../unitree_sdk2_python` or `../repos/unitree_sdk2_python` relative to this project. `./scripts/ensure_unitree_sdk_path.sh` links the first one it finds into `.deps/unitree_sdk2_python`, which is the stable path uv uses. The robot SSH target (for example `unitree@ubuntu`) is separate from this local Python dependency path.

```bash
./scripts/ensure_unitree_sdk_path.sh
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
ssh <robot_ssh_user>@<robot_ssh_host> "cd /home/<robot_ssh_user>/project && ./scripts/setup_vision_server.sh && uv run vision <g1_ip> --no-ssh"
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

## Vision FastAPI stream

For low-latency browser preview, use the headless FastAPI MJPEG server instead of Streamlit. By default it reads the direct RealSense/V4L2 color stream, scales to `640x360`, and serves the latest JPEG at `/stream.mjpg`. YOLO inference is optional and requires `uv sync --only-group vision --only-group train --locked`.

On the Ubuntu vision box, run this once:

```bash
cd ~/Documents/project
./scripts/setup_vision_server.sh
```

Then start the Unitree G1 30 FPS camera stream servers:

```bash
./scripts/run_yolo_stream.sh
```

By default this starts two MJPEG servers:

```text
main camera:  http://0.0.0.0:8000  device=/dev/videohub_pc4
chest camera: http://0.0.0.0:8001  device=/dev/videohub_pc4_ch
```

Open these from the MacBook using the Ubuntu vision box IP, for example:

```text
http://192.168.0.122:8000
http://192.168.0.122:8001
```

The defaults are already set for the current Unitree G1 plan:

```text
main pipeline=v4l2src device=/dev/videohub_pc4 ... framerate=30/1
chest pipeline=v4l2src device=/dev/videohub_pc4_ch ... framerate=30/1
model=none (raw camera stream; set MODEL=... only after installing the train group)
imgsz=320
conf=0.35
infer_every=1
jpeg_quality=60
max_det=20
opencv_threads=16
torch_threads=16 (only used when MODEL is not none)
stream_fps=0 (unbounded; sends each new JPEG immediately)
host=0.0.0.0
port=8000
```

Useful overrides:

```bash
CHEST_PORT=8002 ./scripts/run_yolo_stream.sh
MAIN_DEVICE=video0 CHEST_DEVICE=video1 ./scripts/run_yolo_stream.sh
DUAL_STREAMS=0 ./scripts/run_yolo_stream.sh
PIPELINE="v4l2src device=/dev/video0 ..." ./scripts/run_yolo_stream.sh
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

If `/snapshot.jpg` returns `503`, the camera loop has not produced a decoded frame yet. Check the GStreamer pipeline and make sure nothing drops RTP/H264 packets before `rtph264depay`.

The setup script uses `uv venv --system-site-packages .venv`, so the project keeps the standard `.venv` name while still seeing Ubuntu's system OpenCV with GStreamer enabled. It also removes pip OpenCV wheels because those usually do not include GStreamer.

If YOLO is enabled with `MODEL=...`, trade a little detector update rate for more camera/browser FPS without editing files:

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
uv run loco 192.168.0.212 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
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
uv run python scripts/unitree_loco_minimal.py --robot-ip 192.168.0.4 --loco-service-name ai_sport
uv run python scripts/unitree_loco_minimal.py --interface eno1 --loco-service-name ai_sport
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
uv run loco 192.168.0.212 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
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
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name sport --dds-config-mode no_trace
```

If both probes return `3102` (`Request sending error`) on read-only methods, put the robot into high-level sport/ai-sport mode with the controller and retry. At that point the failure is the robot RPC server not responding on `rt/api/<service>/request`, not DDS initialization. The next useful checks are whether the robot firmware exposes the high-level loco RPC service at all and whether motion mode is enabled on the robot side.

If MotionSwitcher also returns `3102`, first classify the server-to-robot path from the machine that will send commands:

```bash
PYTHONPATH=$PWD/src python3 scripts/robot_eth0_rpc_probe.py \
  --interfaces enP7s7,wlan0 \
  --ping-ip 192.168.0.212 \
  --domain-id 0 \
  --timeout 15.0
```

The probe script sets `PYTHONPATH=./src` for child commands, checks `object_tracking` imports before DDS, and reports per-interface classifications. `--ping-ip` is only a network diagnostic; Unitree DDS discovery uses `--interfaces`, `--domain-id`, and the SDK service names.

Direct server-side read-only checks:

```bash
PYTHONPATH=$PWD/src python3 scripts/g1_loco.py \
  --interface enP7s7 \
  --domain-id 0 \
  --timeout 15.0 \
  --dds-config-mode no_trace \
  diagnose \
  --loco-service-name sport

PYTHONPATH=$PWD/src python3 scripts/g1_loco.py \
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

If `scripts/dds_discovery_probe.py` shows no DDS discovery output on the server interfaces, direct server-side Unitree SDK2 DDS is not reaching the robot. In that state, `probe_loco`, `check_motion_mode`, and `stop_move` from the server will return `3102` because no robot RPC participant is discovered.

Use the robot-side HTTP bridge instead. It keeps DDS local to the robot and lets the external server send ordinary HTTP JSON over Wi-Fi.

On the robot:

```bash
cd ~/humanoid-robot-grasping
git pull
python3 scripts/robot.py serve
```

From the server:

```bash
python3 scripts/robot.py health
python3 scripts/robot.py mode
python3 scripts/robot.py probe
python3 scripts/robot.py stop
```

If `wlan0` does not work inside the bridge, restart it with:

```bash
python3 scripts/robot.py serve --interface eth0
```

The server still talks to `192.168.0.212:8765`; only the robot-local DDS interface changes.

Movement is enabled by default for this robot-local bridge. Use `--read-only` when starting the bridge if you want to disable movement endpoints.

```bash
python3 scripts/robot.py smoke-move
```

Small bounded arm test:

```bash
python3 scripts/robot.py arms-up 0.15
python3 scripts/robot.py forward 0.05 --duration 0.4 --ramp 0.15
python3 scripts/robot.py move --vx 0.05 --vy 0 --omega 0 --duration 0.4 --ramp 0.15
python3 scripts/robot.py stop
```

Avoid interactive keyboard control for now. Use one bounded command at a time (`forward`, `move`, `arms-up`, SDK examples), then send `stop` before the next test.

Unitree SDK example actions can also be listed and run through the same bridge:

```bash
python3 scripts/robot.py sdk-examples
python3 scripts/robot.py sdk-example motion_switcher check_mode
python3 scripts/robot.py sdk-example g1_loco high_stand
python3 scripts/robot.py sdk-example g1_loco move_forward_tiny --speed 0.1 --duration 0.5
python3 scripts/robot.py sdk-example g1_arm_action "hands up"
python3 scripts/robot.py sdk-example g1_arm_action "release arm"
```

The original SDK files these map to are `example/g1/high_level/g1_loco_client_example.py`, `example/g1/high_level/g1_arm_action_example.py`, and `example/motionSwitcher/motion_switcher_example.py`. The wrapper is allowlisted because the original examples are interactive loops and several actions move the robot immediately.

Copied vendor examples are also available verbatim under `scripts/unitree_examples/`. Run these on the robot, not the server:

```bash
cd ~/humanoid-robot-grasping
git pull

python3 scripts/robot.py vendor-example list
python3 scripts/robot.py vendor-example motion_switcher
python3 scripts/robot.py vendor-example g1_loco
python3 scripts/robot.py vendor-example g1_arm_action
python3 scripts/robot.py vendor-example g1_arm5
```

These are the Unitree examples unchanged. For `g1_loco`, type `list` at its prompt, then try IDs from the Unitree menu. For example, ID `3` is Unitree's `move forward` example and ID `5` is `move rotate`.

`amount` is the fraction of the forward arm target. Start around `0.1` to `0.2`. Arm and velocity motion use smoothstep easing so they ease in/out instead of snapping to a linear ramp.

Only use robot-local checks to isolate low-level robot networking after server-side tests are exhausted:

```bash
ssh unitree@192.168.0.212
cd ~/humanoid-robot-grasping
git pull
python3 scripts/robot_eth0_rpc_probe.py --interfaces eth0,wlan0 --ping-ip 192.168.123.1
```

If the robot does not show an obvious `ai_sport`/`loco` Linux service to start manually, use the SDK motion switcher path:

```bash
uv run loco 192.168.0.212 check_motion_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.212 select_ai_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
```

`check_motion_mode` is read-only. `select_ai_mode` calls `MotionSwitcherClient.SelectMode("ai")`, so only run it when the robot is physically safe and the controller/e-stop is ready.

### Robot-local shoulder pitch arm test

This script runs on the robot itself and uses low-level SDK2 topics. It releases high-level motion mode, reads the current full-body posture from `rt/lowstate`, holds every motor at that posture, and only offsets the selected shoulder pitch joint(s).

On the robot:

```bash
cd ~/humanoid-robot-grasping
git pull

python3 scripts/robot.py shoulder-pitch \
  --side both \
  --sign 1 \
  --delta 0.25 \
  --ramp-seconds 3.0 \
  --hold-seconds 5.0
```

Dry-run without DDS or movement:

```bash
python3 scripts/robot.py shoulder-pitch --smoke --side both --sign 1 --delta 0.25
```

If the shoulder pitch direction is backwards, retry with `--sign -1` or `--direction negative`. If left/right need opposite directions, use `--left-sign 1 --right-sign -1` or the reverse. Start with a smaller `--delta 0.1` if you only want a small motion check.

After a read-only probe returns `ok: true`, a tiny movement smoke test is available but guarded:

```bash
uv run loco 192.168.0.212 smoke_move \
  --interface enP7s7 \
  --loco-service-name sport \
  --dds-config-mode no_trace
```
