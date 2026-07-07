# Unitree SDK2 Notes

`unitree_sdk2` is integrated through the SDK's G1 C++ loco example binary because this checkout does not include Python bindings.

## Normal Usage

Use the robot IP. The project runs `ip route get <robot_ip>` and passes the resolved local interface into SDK2.

```bash
uv run g1-loco 192.168.123.161 get_fsm_id
uv run g1-loco 192.168.123.161 stand_up
uv run g1-loco 192.168.123.161 move --velocity "0.2 0 0 1.0"
uv run g1-loco 192.168.123.161 stop_move
```

Manual override is still available:

```bash
uv run g1-loco --network-interface enP7s7 get_fsm_id
```

## SDK Helper

The wrapper defaults to:

```text
~/Documents/unitree_sdk2
```

It looks for a built G1 loco binary at:

```text
~/Documents/unitree_sdk2/build/bin/g1_loco_client
~/Documents/unitree_sdk2/build/g1_loco_client
```

Build it with:

```bash
cd ~/Documents/unitree_sdk2
mkdir -p build
cd build
cmake ..
make g1_loco_client
```

## Vision Usage

For local webcam testing:

```bash
uv run python scripts/run_manual_tracker.py --camera 0 --output runs/vision_test
```

For G1 camera testing after the robot is reachable:

```bash
uv run g1-vision 192.168.123.161
```

If the actual camera URL is known, use it directly:

```bash
uv run g1-vision 192.168.123.161 --camera-url "rtsp://{ip}:8554/live"
```

The SDK2 checkout inspected here does not contain a G1-specific vision/video client. `g1-vision` therefore uses OpenCV stream URLs and tries common RTSP-style conventions unless `--camera-url` or `G1_CAMERA_URL` is provided.
