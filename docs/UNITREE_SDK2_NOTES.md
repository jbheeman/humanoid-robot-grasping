# Unitree SDK2 Notes

Install the loco extra to use this:

```bash
uv sync --extra loco
```

(`unitree_sdk2py` is the Python module import path.)

No local `unitree_sdk2` C++ checkout is required for this project.

Vision and loco both resolve the local interface from:

```bash
ip route get <robot_ip>
```

### Visual Probe

Use this first from the machine on the robot network:

```bash
uv run vision <g1_ip> --sdk2-video --sdk2-timeout 3
```

This resolves the route/interface, initializes SDK2 DDS, calls `videohub.GetImageSample`, decodes the returned image bytes, and writes `runs/vision_test/snapshot.jpg`.

### Notes for Vision

- The current fallback camera candidates include UDP/GStreamer pipelines listening locally on port `5600`, but this is not a confirmed G1 default.
- Do not bind `udpsrc address=` to the robot IP. Use `G1_CAMERA_BIND_ADDRESS=<local_interface_ip>` only when the receiver must bind a specific local address.
- `opencv-python` commonly reports no GStreamer support; check `uv run vision <robot_ip> --diagnose` and look for `opencv.gstreamer: true`.
- If your stream is RTSP/SSH-only, pass `--camera-url` or `--no-ssh`/SSH flags manually.
- If no gstreamer plugin support exists locally, install/OpenCV with GStreamer.
- Vision also accepts positional SSH credentials: `vision <robot_ip> <user@host>` or `vision <robot_ip> <user> <host>`.

### Notes for Loco

- `uv run loco <robot_ip> stand_up`
- `uv run loco <robot_ip> move --velocity "vx vy omega [duration]"`
- `uv run loco <robot_ip> --diagnose`
