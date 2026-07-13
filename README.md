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
        │ ROS 2 / CycloneDDS                     │ UDP 5600
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
- `ros-foxy-rmw-cyclonedds-cpp`
- `ros-foxy-rosidl-generator-dds-idl`
- `python3-colcon-common-extensions`, Git, `uv`, GStreamer, and the depth driver

GB10 prerequisites:

- Ubuntu 24.04 with `/usr/bin/python3` 3.12
- ROS 2 Jazzy at `/opt/ros/jazzy`
- `ros-jazzy-rmw-cyclonedds-cpp`
- `ros-jazzy-rosidl-generator-dds-idl`
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

Use the real IP reachable from the other machine. The launchers generate
CycloneDDS static-peer configuration and default to ROS domain `0`.

On the robot:

```bash
uv run g1 robot start --client-ip <GB10_IP>
```

On the GB10:

```bash
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

Open the single UI:

```text
http://<GB10_IP>:8000/
```

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

- `/g1/arm/target` (`g1_control_interfaces/msg/ArmTarget`)
- `/g1/arm/state` (`g1_control_interfaces/msg/ArmState`)
- `/g1/depth` (`g1_control_interfaces/msg/CompressedDepth`)
- `/g1/arm/control` (`g1_control_interfaces/srv/ArmControl`)
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
