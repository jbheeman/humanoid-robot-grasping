# Project commands

`g1` is the operator entry point; host-specific scripts remain stable
automation entry points.

```bash
uv run g1 --help
uv run g1 <group> <workflow> --help
```

## Setup and startup

```bash
# First bootstrap on each host
bash scripts/robot/setup.sh
bash scripts/gb10/setup.sh

# Normal operation
uv run g1 robot start  # defaults to GB10 192.168.0.66
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run

# GB10 only: install official UnifoLM and open the typed inference terminal
uv run g1 setup vla
uv run g1 gb10 vla
```

The VLA terminal is inference-only unless `--execute` is supplied. For a later
guarded hardware test, run the GB10 dashboard with `--dry-run --vla-preview` so
it subscribes to RGB/depth/arm state but creates no command publisher. Then run
`uv run g1 gb10 vla --execute --max-waypoints 3 --motion-period 0.15` in a
second terminal. The robot launcher must independently include
`--allow-movement --expected-motion-mode ai` and the validated calibration.

Useful launch options:

```bash
uv run g1 robot start \
  --client-ip <GB10_IP> --interface wlan0 --control-peer 192.168.123.1 \
  --ros-domain-id 42 --calibration /secure/g1-camera.yaml

uv run g1 gb10 start \
  --robot-host <ROBOT_IP> --ros-interface auto --ros-domain-id 42 \
  --calibration /secure/g1-camera.yaml --dry-run
```

Use `--execute` on GB10 only after commissioning. The robot must also be
launched with explicit movement permission and an expected mode; neither side
can enable motion alone.

The UI and commissioning page are:

```text
http://<GB10_IP>:8000/
http://<GB10_IP>:8000/commissioning/
```

## Commissioning

```bash
# Robot: read-only
CLIENT_IP=<GB10_IP> G1_ROBOT_ID=g1-lab-01 \
  uv run g1 arm commissioning

# Robot: movement-capable only after the read-only preflight
CLIENT_IP=<GB10_IP> G1_ROBOT_ID=g1-lab-01 \
EXPECTED_MOTION_MODE=ai ALLOW_MOVEMENT=1 \
  uv run g1 arm commissioning
```

There is no commissioning token or robot URL. Operate through the GB10 page.

## Read-only diagnostics

```bash
uv run g1 inspect cameras
uv run g1 inspect ros --role robot --peer <GB10_IP> --interface wlan0
uv run g1 inspect ros --role gb10 --peer <ROBOT_IP>
uv run g1 robot scan --help
uv run g1 robot loco --help
```

`g1 inspect dds`, `g1 robot command`, robot port flags, token flags, depth
WebSocket URLs, and SDK backend flags describe the removed transport and are
not valid commands.

## Perception, research, and calibration

```bash
uv run g1 vision snapshot <robot-ip> --diagnose
uv run g1 tune joint-audit
uv run g1 tune analyze runs/research/arm_tracking
uv run g1 tune compare RUN_A RUN_B
uv run g1 calibrate camera --help
```

## Data and training

```bash
uv run g1 data install
uv run g1 data augment --help
uv run g1 data capture --help
uv run g1 train detector --help
uv run g1 train evaluate --help
uv run g1 train guarded
```

Never run training commands or install CUDA dependencies on the robot.
