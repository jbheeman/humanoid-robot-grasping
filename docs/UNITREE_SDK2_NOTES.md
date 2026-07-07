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

### Notes for Vision

- Default camera candidates for G1 are UDP/GStreamer pipelines using port `5600`.
- If your stream is RTSP/SSH-only, pass `--camera-url` or `--no-ssh`/SSH flags manually.
- If no gstreamer plugin support exists locally, install/OpenCV with GStreamer.
- Vision also accepts positional SSH credentials: `vision <robot_ip> <user@host>` or `vision <robot_ip> <user> <host>`.

### Notes for Loco

- `uv run loco <robot_ip> stand_up`
- `uv run loco <robot_ip> move --velocity "vx vy omega [duration]"`
- `uv run loco <robot_ip> --diagnose`
