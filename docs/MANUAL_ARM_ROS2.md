# G1 manual arm control over ROS 2

This is the movement-only path used before plush tracking. The GB10 publishes
absolute seven-joint targets on project ROS domain 42. One robot-local process
merges both sides into Unitree's required 14-joint frame and is the only
publisher to native `rt/arm_sdk` on domain 0.

The arm-only launcher starts no camera or depth stream. The regular robot and
GB10 launchers can include this bridge with `--arm-commissioning`, and the
GB10 client also supports bounded relative Cartesian IK tests.

## Topics

- `/g1/arm_control/left/command` — `g1_control_interfaces/msg/ArmSideTarget`
- `/g1/arm_control/right/command` — `g1_control_interfaces/msg/ArmSideTarget`
- `/g1/arm_control/heartbeat` — session deadman
- `/g1/arm_control/status` — typed bridge state plus complete JSON report
- `/g1/arm_control/joint_states` — measured 14-arm joint state
- `/g1/arm_control/request` and `/response` — correlated enable/stop/state RPC

The control RPC uses topics instead of a custom ROS service because this has
proven interoperable between the robot's Foxy/Fast DDS process and the GB10's
Jazzy/Fast DDS process.

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

To run camera, depth, and the manual arm bridge from one launcher per host,
stop the old arm-only bridge and use:

```bash
# Robot
CLIENT_IP=192.168.0.66 scripts/robot/start.sh --arm-commissioning

# GB10
scripts/gb10/start.sh --arm-commissioning
```

This mode permits movement but always starts `DISARMED`; only a fresh client
session with a heartbeat can temporarily take arm ownership.

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

Manual delta signs are inverted only at this operator-facing boundary to match
the observed forward/back convention. Canonical Unitree/URDF angles used by IK
are not modified. This example moves two joints together, returns to the
measured baseline, and releases arm weight to zero:

```bash
scripts/gb10/arm-remote.sh move \
  --side right \
  --joint-delta right_shoulder_pitch_joint=0.12 \
  --joint-delta right_elbow_joint=0.12 \
  --duration 2 \
  --hold 5
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

## First relative Cartesian IK test

The IK frame is the G1 URDF pelvis/waist-root frame: +X forward, +Y left, and
+Z up. This command solves from fresh measured right-arm joints, requests the
hand 1 cm upward while preserving its measured orientation, moves all required
joints through guarded 0.05-rad increments, verifies the measured Cartesian
result, returns, and releases:

```bash
scripts/gb10/arm-remote.sh ik --dz 0.01 --duration 2 --hold 3
```

Start with +Z because the normal hanging hand is beside the hip; a direct +X
move from that pose can correctly fail the swept-collision check. Perception
never publishes Cartesian targets or native Unitree DDS packets directly: the
GB10 solves IK to seven canonical joint angles and the robot-local bridge owns
interpolation, heartbeat release, state gates, and `rt/arm_sdk` output.

## Compare the two Unitree controller profiles

The robot bridge exposes two deliberate A/B candidates:

- `sdk2`: the installed G1 arm7 SDK2 example profile, 50 Hz with Kp 60 and
  Kd 1.5 on all arm joints.
- `xr`: the current Unitree XR profile, 250 Hz with Kp/Kd 80/3 on the four
  shoulder/elbow joints and 40/1.5 on the three wrist joints.

Select a profile when starting the robot bridge. Restarting is required to
change profiles, and startup itself does not arm or move the robot:

```bash
# Candidate A
MANUAL_ARM_PROFILE=sdk2 CLIENT_IP=192.168.0.66 \
  scripts/robot/manual-arm.sh move

# Candidate B (after stopping candidate A)
MANUAL_ARM_PROFILE=xr CLIENT_IP=192.168.0.66 \
  scripts/robot/manual-arm.sh move
```

Run the same user-authorized GB10 movement for each candidate and save its
measured response. Return the robot to the same preparation pose first:

```bash
scripts/gb10/arm-remote.sh ik \
  --dz 0.01 --duration 2 --hold 3 \
  --trace-output runs/arm/ik-z10-sdk2.json
```

Use a distinct trace filename for the XR trial. Compare Cartesian endpoint
error, movement in the two unintended axes, overshoot, settling, and any
following-error fault; visual appearance alone is not the selection metric.
