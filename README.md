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

## Grasping (arms and hands only)

`grasp` drives *only* the G1 arm joints (indices 15-28) over the low-level
`rt/arm_sdk` channel and the Dex3-1 hands over `rt/dex3/{left,right}/cmd`. The
legs and waist stay under the balance controller, so the robot keeps standing
while the arms reach out, close the hands, and hold an object steadily.

Requires the loco extra:

```bash
uv sync --extra loco
```

Stand the robot up first, then run the grasp:

```bash
uv run loco 192.168.0.4 stand_up
uv run loco 192.168.0.4 balance_stand
uv run grasp 192.168.0.4
```

The sequence is: engage arms → move to a ready pose → reach forward → close the
hands → hold firmly, then release the hands and return the arms home on exit.

Validate safely before commanding motion:

```bash
uv run grasp 192.168.0.4 --check      # print live arm joint angles, no motion
uv run grasp 192.168.0.4 --dry-run    # log the planned sequence, no motion
```

Commonly useful options:

```bash
uv run grasp 192.168.0.4 --hold-forever          # hold until Ctrl+C, then release
uv run grasp 192.168.0.4 --hold-seconds 20 --lift # hold 20s and lift slightly
uv run grasp 192.168.0.4 --hand none              # arms only (no Dex3-1 hands)
uv run grasp 192.168.0.4 --side left              # one arm/hand
uv run grasp 192.168.0.4 --kp 80 --kd 2.0         # firmer hold
uv run grasp 192.168.0.4 --yes                    # skip the safety countdown
```

Events are logged to `runs/grasp/grasp.jsonl`.

Notes:
- The arm poses in `DEFAULT_POSES` and the Dex3-1 open/close vectors in
  `src/object_tracking/g1_grasp.py` are starting points — **tune them on your
  hardware** (joint sign conventions and reachable ranges vary per unit).
- Keep a hand on the e-stop; a `--countdown` runs before any motion.
