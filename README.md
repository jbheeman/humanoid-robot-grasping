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

## Unitree SDK2 loco

Run commands with robot IP:

```bash
uv run loco 192.168.0.4 stand_up
uv run loco 192.168.0.4 balance_stand
uv run loco 192.168.0.4 move --velocity "0.2 0 0 1.0"
uv run loco 192.168.0.4 stop_move
uv run loco 192.168.0.4 --diagnose
```

`--network-interface` is only needed if route detection fails.
