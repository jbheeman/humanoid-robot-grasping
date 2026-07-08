# Humanoid Robot Grasping

## Setup

From the project root:

```bash
git switch aarav
uv sync
```

This installs vision plus the Unitree SDK2 Python runtime. `uv sync --extra loco` is still accepted for the old workflow, but the SDK is now a normal dependency so plain `uv sync` will not prune it:

```bash
uv sync --extra loco
```

Loco commands are run from the server that is connected to the robot network, not necessarily on the robot itself. Pass the robot LAN IP to `uv run loco`. The local editable `unitree-sdk2py==1.0.1` checkout must exist on that server at `../unitree_sdk2_python` relative to this project. For example, if the server checkout is `/home/neel/humanoid-robot-grasping`, uv expects the SDK at `/home/neel/unitree_sdk2_python`. The robot SSH target (for example `unitree@ubuntu`) is separate from this local Python dependency path.

```bash
uv sync --extra loco
uv run loco <robot_lan_ip> stop_move
```

## Vision runbook

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

If reachable, this opens one frame and saves:
- `runs/vision_test/snapshot.jpg`
- `runs/vision_test/camera.json`

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
uv run vision 192.168.0.4 --camera-url "udpsrc address={ip} port=5600 ! application/x-rtp,media=(string)video ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
uv run vision 192.168.0.4 --no-ssh
uv run vision 192.168.0.4 --show
```

Notes:
- G1 default transport is GStreamer/UDP (`gstreamer` mode), not RTSP.
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
