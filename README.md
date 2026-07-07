# Humanoid Robot Grasping

## Vision Quick Start

For a local webcam or USB camera:

```bash
git switch aarav
uv sync
uv run python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
```

For a G1 camera stream after the robot is powered on and reachable:

```bash
uv run g1-vision 192.168.123.161
```

If the G1 camera uses a known URL, pass it directly:

```bash
uv run g1-vision 192.168.123.161 --camera-url "rtsp://{ip}:8554/live"
```

In the first video frame, drag a box around the moving plush object and press Enter. Press `q` or Escape to stop.

The run writes:

- `runs/object_manual_test/tracking.mp4`
- `runs/object_manual_test/tracking.jsonl`

## Unitree G1 SDK2 Commands

Robot commands can be addressed by robot IP. The project resolves the local network interface automatically.

Build the SDK helper once:

```bash
cd ~/Documents/unitree_sdk2
mkdir -p build
cd build
cmake ..
make g1_loco_client
```

Then run commands from this project:

```bash
uv run g1-loco 192.168.123.161 get_fsm_id
uv run g1-loco 192.168.123.161 stand_up
uv run g1-loco 192.168.123.161 move --velocity "0.2 0 0 1.0"
uv run g1-loco 192.168.123.161 stop_move
```

Only use `--network-interface` if automatic route detection fails.
