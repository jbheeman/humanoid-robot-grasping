# Humanoid Robot Grasping

Perception-guided Unitree G1 arm control with ROS 2 between the robot and the
GB10. The robot no longer runs project HTTP/WebSocket control servers. The
only web server is the GB10 browser UI on port `8000`; RGB remains an H264 RTP
relay over UDP because it is already efficient and carries no control command.

## Runtime layout

```text
G1 robot (Ubuntu 20.04, ROS 2 Foxy, Python 3.8)
  /lowstate + Unitree API topics
  guarded arm/depth ROS node
  H264 RGB relay ────────────────────────────────┐
        │ ROS 2 / Fast DDS                       │ UDP 5600
        ▼                                        ▼
GB10 (Ubuntu 24.04, ROS 2 Jazzy, Python 3.12)
  depth fusion + YOLO + IK + research + browser UI :8000
        │ HTTP (UI only)
        ▼
Browser
```

The application publishes high-level seven-joint targets to the robot. Only
the guarded robot node publishes Unitree `/arm_sdk` commands. ROS 2 still uses
DDS underneath, but project code does not construct DDS channels or expose a
DDS-based control backend.

## First-time setup

Clone the same revision on both machines. Setup validates rather than silently
installing system packages.

Robot prerequisites:

- Ubuntu 20.04 with `/usr/bin/python3` 3.8
- ROS 2 Foxy at `/opt/ros/foxy`
- `ros-foxy-rmw-fastrtps-cpp`
- `ros-foxy-rosidl-generator-dds-idl`
- `python3-colcon-common-extensions`, Git, `uv`, GStreamer, and the depth driver

GB10/replacement workstation prerequisites:

- Ubuntu 24.04/Python 3.12 with ROS 2 Jazzy, or Ubuntu 22.04/Python 3.10 with ROS 2 Humble
- `rmw-cyclonedds-cpp`, `rmw-fastrtps-cpp`, and `rosidl-generator-dds-idl` for that distro
- `python3-colcon-common-extensions`, Git, `uv`, GStreamer, and NVIDIA support

Run on the matching host:

```bash
# Robot bootstrap (also works before the g1 command is installed)
bash scripts/robot/setup.sh

# GB10 bootstrap
bash scripts/gb10/setup.sh
```

Both scripts pin Unitree ROS 2 v0.3.0 at commit
`12c080cb91ee55854358ee9413c2e36e543c36ee`, then build its `unitree_hg` and
`unitree_api` packages and the tracked `ros_ws/src/g1_control_interfaces`
package. Generated workspaces live under `.ros/<distro>/` and are sourced by
the launchers automatically.

## Normal startup

Use the real IP reachable from the other machine. ROS 2 works over the Wi-Fi
LAN; Ethernet is not required. Both launchers use Fast DDS, explicit peers,
and ROS domain `42` by default.

On the robot:

```bash
uv run g1 robot start --client-ip <GB10_IP> \
  --calibration /home/unitree/.config/g1-grasping/g1-tabletop-calibration.json
```

On the GB10:

```bash
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

### Make robot observations persistent

For a lab G1 that should always provide camera and read-only ROS observations
after boot, install the observation bridge once on the robot:

```bash
bash scripts/robot/install-observation-bridge-service.sh
```

It starts the native RGB relay and robot observation node for GB10
`192.168.0.66`, remains disarmed, and never sets movement permission. After
that one-time install, normal VLA preview on the GB10 needs only:

```bash
uv run g1 gb10 start --robot-host 192.168.0.213 --dry-run --vla-preview
```

To remove it later, run
`bash scripts/robot/uninstall-observation-bridge-service.sh` on the robot.

Open the single UI:

```text
http://<GB10_IP>:8000/
```

### UnifoLM-VLA terminal (GB10 only)

Install the pinned official Unitree runtime and model on the GB10, then open
the typed task terminal:

```bash
uv run g1 setup vla
uv run g1 gb10 vla
```

To load the newest completed checkpoint under `models/unifolm_vla` instead of
the Unitree base model:

```bash
bash scripts/gb10/vla-trained.sh
```

Set `VLA_CHECKPOINT=/absolute/path/to/final_model/pytorch_model.pt` to select a
specific run. The launcher remains inference-only unless `--execute` is also
supplied.

The default terminal is inference-only: it reads the latest unannotated RGB
frame plus measured arm state and prints the proposed 23D action chunk without
publishing a command. Type a task such as `raise the right hand slightly`;
`/status`, `/help`, and `/exit` are built in.

After offline validation and with a spotter/e-stop, guarded execution uses a
command-free dashboard transport so the VLA is the sole ROS command publisher:

```bash
# Robot: explicit permission is required; startup remains DISARMED.
uv run g1 robot start --client-ip 192.168.0.66 \
  --calibration /home/unitree/.config/g1-grasping/g1-tabletop-calibration.json \
  --allow-movement --expected-motion-mode ai

# GB10 terminal 1: observations only; never publishes an arm command.
uv run g1 gb10 start --robot-host 192.168.0.213 --dry-run --vla-preview \
  --calibration runs/localization/g1-tabletop-calibration.json

# GB10 terminal 2: the only project command publisher.
uv run g1 gb10 vla --execute --max-waypoints 3 --motion-period 0.15
```

Execution first plans and streams a collision-checked clearance pose. Within
the calibrated table footprint, checked links must remain at least 5 cm above
the calibrated tabletop plane. It then takes a fresh observation, re-runs the
policy, and permits at most three right-arm translation waypoints with a 0.025
rad per-joint step cap. Left-arm, waist, orientation, and gripper predictions
are ignored. Any stale state, failed keepalive, calibration change, incomplete
table footprint, IK failure, or collision check stops the session.

The base policy was trained with head and wrist cameras and task-specific G1
demonstrations. A single head D435I can exercise inference, but reliable plush
contact requires a matching XR-teleop dataset and offline/simulation evaluation
before connecting proposals to the guarded arm controller. The robot's lack of
a controllable hand also means the practical initial goal is palm contact, not
a verified grasp.

Commissioning is at `http://<GB10_IP>:8000/commissioning/`. There are no
bearer-token files, robot HTTP ports, separate static viewer, or robot SSH
tunnels. If the browser is off-LAN, tunnel only GB10 port `8000`:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@GB10_IP
```

Lab-specific defaults remain available through:

```bash
bash scripts/robot/lab-start.sh
bash scripts/gb10/lab-start.sh
```

Override `CLIENT_IP`, `ROBOT_HOST`, `ROBOT_INTERFACE`, `ROS_INTERFACE`, or
`ROS_DOMAIN_ID` only when the network layout changes. Robot discovery also
includes `UNITREE_CONTROL_PEER`, defaulting to `192.168.123.1`.

## ROS interfaces

Unitree contracts used on the robot:

- `/lowstate` (`unitree_hg/msg/LowState`)
- `/arm_sdk` (`unitree_hg/msg/LowCmd`)
- `/api/motion_switcher/{request,response}`
- `/api/{sport|ai_sport}/{request,response}`

Project contracts shared by Foxy and Jazzy:

- `/g1/arm/target_json` (`std_msgs/msg/String`, validated JSON envelope)
- `/g1/arm/state_json` (`std_msgs/msg/String`, validated JSON report)
- `/g1/depth` (`g1_control_interfaces/msg/CompressedDepth`)
- `/g1/arm/control/{request_json,response_json}` (`std_msgs/msg/String`)
- `/g1/commissioning/command`
  (`g1_control_interfaces/srv/CommissioningCommand`)
- `/g1/commissioning/state`
  (`g1_control_interfaces/msg/CommissioningState`)

Inspect discovery without sending a command:

```bash
# On robot; peer is the GB10
uv run g1 inspect ros --role robot --peer <GB10_IP> --interface wlan0

# On GB10; peer is the robot
uv run g1 inspect ros --role gb10 --peer <ROBOT_IP>
```

`g1 inspect dds`, SDK backend flags, robot HTTP ports, depth WebSocket URLs,
and token flags are intentionally removed.

## Arm safety and commissioning

Robot startup is disarmed. Starting the GB10 in execute mode does not bypass
the robot safety controller. Movement requires fresh state, stable standing,
the expected motion mode, exclusive publisher ownership, valid calibration,
joint/slew/acceleration/following-error limits, and a live heartbeat. Link
loss performs a bounded release to zero arm weight.

Commission in this order:

1. Start the robot commissioning node read-only.
2. Start the GB10 dry-run UI and verify ROS health and joint ordering.
3. With a cleared area, spotter, physical e-stop, and the robot supported,
   restart commissioning with explicit movement permission.
4. Use one `0.01` rad joint jog at a time; verify sign and measured following
   error after every jog.
5. Verify stop and the bounded zero-weight release before teaching or
   promoting a home pose.

See [the commissioning runbook](docs/ARM_COMMISSIONING_RUNBOOK.md) and
[the tracking runbook](docs/ARM_TRACKING_RUNBOOK.md) for exact commands.

## What to do immediately after the rewrite

1. Build the interfaces under Foxy and Jazzy and run the repository tests.
2. Run `g1 inspect ros` on both hosts; confirm `/lowstate`, project state, and
   depth arrive before enabling any publisher.
3. Run both launchers in read-only/dry-run mode and test link loss, stale
   targets, node exit, duplicate publishers, and GB10 shutdown.
4. Perform guarded `0.01` rad commissioning jogs on real hardware.
5. Only after commissioning passes, enable tracking and then run the smallest
   bounded locomotion smoke test.
6. Resume perception/IK tuning from recorded real telemetry.

Simulation is not required for normal arm movement after commissioning. Use
the `sim` branch for broad gain, velocity, and acceleration searches and for
regression testing before riskier parameter changes. Its welded-pelvis,
contact-disabled MuJoCo setup cannot validate walking, balance recovery,
collisions, or grasp contact. Introduce simulated gains one at a time through
the guarded physical jog workflow.

## Vision, research, and training

The GB10 UI exposes RGB, depth, detections, tracks, prediction, IK/arm state,
health, and research exports. Common workflows remain under the unified CLI:

```bash
uv run g1 tune joint-audit
uv run g1 tune analyze runs/research/arm_tracking
uv run g1 data install
uv run g1 data augment --help
uv run g1 train detector --help
uv run g1 train evaluate --help
```

Training outputs stay under `models/`; run telemetry stays under
`runs/research/`. Never install CUDA or training dependencies in the robot
runtime.

## Source and operator references

- [Architecture](docs/ARCHITECTURE.md)
- [Command reference](docs/COMMANDS.md)
- [Lab launchers](docs/LAB_LAUNCHERS.md)
- [Arm commissioning](docs/ARM_COMMISSIONING_RUNBOOK.md)
- [Arm tracking](docs/ARM_TRACKING_RUNBOOK.md)
- [Tuning guide](docs/TUNING_GUIDE.md)
- [Third-party sources](docs/THIRD_PARTY_SOURCES.md)
