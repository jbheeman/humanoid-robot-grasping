# G1 manual arm control over ROS 2

This is the movement-only path used before plush tracking. The GB10 publishes
absolute seven-joint targets on project ROS domain 42. One robot-local process
merges both sides into Unitree's required 14-joint frame and is the only
publisher to native `rt/arm_sdk` on domain 0.

It does not start a camera, depth stream, localization, IK, or the browser
commissioning UI.

## Topics

- `/g1/arm_control/left/command` — `g1_control_interfaces/msg/ArmSideTarget`
- `/g1/arm_control/right/command` — `g1_control_interfaces/msg/ArmSideTarget`
- `/g1/arm_control/heartbeat` — session deadman
- `/g1/arm_control/status` — typed bridge state plus complete JSON report
- `/g1/arm_control/joint_states` — measured 14-arm joint state
- `/g1/arm_control/request` and `/response` — correlated enable/stop/state RPC

The control RPC uses topics instead of a custom ROS service because this has
proven interoperable between the robot's Foxy/Fast DDS process and the GB10's
Jazzy/CycloneDDS process.

## Install or rebuild after pulling

The new message types must be generated separately on both ROS distributions.

On the robot:

```bash
cd ~/humanoid-robot-grasping
scripts/robot/setup.sh
```

On the GB10:

```bash
sudo apt-get install ros-jazzy-rmw-fastrtps-cpp
cd ~/Documents/project
scripts/gb10/setup.sh
```

The manual arm channel uses Fast DDS on both hosts. The stock robot's Foxy
Fast DDS process eventually aborts while parsing repeated discovery/type data
from a Jazzy/Cyclone participant; using the same RMW on this isolated channel
avoids that failure. The regular GB10 camera/tracking launcher can continue to
use CycloneDDS.

## Start the robot bridge

Stop other project arm bridges and direct SDK2 arm examples first. With a
spotter, clear workspace, and physical stop available, run on the robot:

```bash
cd ~/humanoid-robot-grasping
CLIENT_IP=192.168.0.66 scripts/robot/manual-arm.sh move
```

Use `observe` instead of `move` to verify ROS without permitting motor output.

## Verify from GB10 without moving

```bash
cd ~/Documents/project
scripts/gb10/arm-remote.sh inspect
uv run g1 inspect ros --role gb10 --peer 192.168.0.213 --profile manual
```

The bridge must report motion mode `ai`, stable standing, fresh LowState,
verified arm ownership, and healthy motor status. `inspect` never enables the
arm.

## First custom movement test

This command latches the measured pose, ramps Unitree arm weight, moves only
the right shoulder pitch by +0.05 rad over two seconds, holds for 0.5 seconds,
returns to the measured baseline, and releases arm weight to zero:

```bash
scripts/gb10/arm-remote.sh move \
  --side right \
  --joint right_shoulder_pitch_joint \
  --delta 0.05 \
  --duration 2
```

The robot rejects a step above 0.05 rad, stale/replayed messages, the wrong
joint order, guarded joint-limit violations, fast trajectories, stale
LowState, unexpected motion mode, unstable stance, another arm publisher, or
unhealthy motors. Loss of the GB10 heartbeat freezes the interpolated target
and ramps arm ownership to zero.

Use this motion-free command to stop, release, and clear a latched fault:

```bash
scripts/gb10/arm-remote.sh stop
```

After this shoulder cycle passes, repeat on the left side. Plush XYZ and IK
targets should later publish through these same side topics; perception never
publishes directly to Unitree DDS.
