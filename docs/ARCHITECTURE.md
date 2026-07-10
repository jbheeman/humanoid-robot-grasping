# Runtime architecture

The project uses three intentionally separate roles. The roles share the
repository and `uv run g1` interface, but they do not share a Python runtime or
publish DDS across the network.

```text
Unitree G1 robot
  RGB multicast relay + RealSense depth service + local arm DDS bridge
             │ HTTP/WebSocket/RTP over the trusted robot network
             ▼
GB10 workstation
  GStreamer decode + YOLO + depth fusion + IK + FastAPI + browser viewer
             │ SSH tunnel or trusted LAN HTTP
             ▼
MacBook browser
```

## Canonical commands

Run these from the repository root on the machine that owns the role:

```bash
uv run g1 setup robot
uv run g1 robot start --client-ip <GB10_IP> --token-file <TOKEN>

uv run g1 setup gb10
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run

uv run g1 setup local
uv run g1 local start --source realsense --device /dev/video0
```

The robot startup is disarmed by default. Arm commissioning is a separate
explicit workflow:

```bash
uv run g1 arm commissioning
```

## Ownership and data flow

- `scripts/robot/` contains robot-local services and the only process that
  publishes `rt/arm_sdk` DDS commands.
- `scripts/gb10/` contains inference, fusion, IK, research telemetry, and the
  static browser viewer. The GB10 sends high-level HTTP targets to the robot
  arm bridge; it does not publish Unitree DDS.
- `scripts/local/` contains lightweight camera, OpenCV, RealSense, and replay
  testing. It is not required for the robot or GB10 deployment.
- `scripts/data/`, `scripts/training/`, and `scripts/dev/` contain offline
  workflows and diagnostics and never start robot movement implicitly.
- `scripts/vendor/` contains copied Unitree examples and is kept separate from
  repository-native runtime code.

The MacBook does not need a display server on the robot or GB10. It opens the
URL printed by the GB10 launcher, optionally through SSH port forwarding.

## Ports

| Role | Port | Purpose |
| --- | ---: | --- |
| Robot | 8766 | Authenticated arm bridge |
| Robot | 8767 | Hardware depth WebSocket service |
| GB10 | 8000 | FastAPI stream, health, tracking, and research API |
| GB10 | 8080 | Static browser viewer |
| Robot → GB10 | UDP 5600 | Compressed H264 RGB relay |

Keep the robot and GB10 services on the trusted lab network. Use SSH tunnels
when the MacBook is not on that network; do not expose the arm bridge directly
to an untrusted interface.
