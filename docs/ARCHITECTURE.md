# Runtime architecture

The deployment has three roles and one browser-facing server.

```text
Unitree G1 / Foxy / Python 3.8
  official Unitree ROS topics
  guarded arm + commissioning + depth node
  H264 RTP relay ─────────────────────────────┐
       │ ROS 2 over CycloneDDS                │ UDP 5600
       ▼                                      ▼
GB10 / Jazzy / Python 3.12
  ROS client + YOLO + fusion + IK + research + FastAPI UI
       │ HTTP :8000 (UI and UI-local APIs only)
       ▼
Browser
```

## Ownership and safety boundary

- `scripts/robot/ros_node.py` is the only project process allowed to publish
  `/arm_sdk`. It subscribes `/lowstate`, owns the 250 Hz guarded controller,
  publishes compressed depth, and provides ROS commissioning services.
- The GB10 publishes only high-level seven-joint `ArmTarget` messages. Its UI
  routes call the local GB10 ROS client; they are not remote robot HTTP calls.
- The safety controller stays on the robot. A GB10 crash, stale target, ROS
  link loss, or heartbeat loss causes a bounded release.
- Locomotion uses Unitree ROS request/response topics. It has no daemon or
  persistent HTTP server.
- RGB remains RTP/H264 over UDP. It is a data-only stream and is independent
  from the ROS control deadman.

## ROS graph

Robot-local Unitree interfaces:

| Name | Type/direction |
| --- | --- |
| `/lowstate` | `unitree_hg/msg/LowState` → robot node |
| `/arm_sdk` | robot node → `unitree_hg/msg/LowCmd` |
| `/api/motion_switcher/{request,response}` | Unitree RPC topics |
| `/api/{sport|ai_sport}/{request,response}` | Unitree locomotion RPC topics |

Cross-host project interfaces:

| Name | Type |
| --- | --- |
| `/g1/arm/target` | `g1_control_interfaces/msg/ArmTarget` |
| `/g1/arm/state` | `g1_control_interfaces/msg/ArmState` |
| `/g1/depth` | `g1_control_interfaces/msg/CompressedDepth` |
| `/g1/arm/control` | `g1_control_interfaces/srv/ArmControl` |
| `/g1/commissioning/command` | `g1_control_interfaces/srv/CommissioningCommand` |
| `/g1/commissioning/state` | `g1_control_interfaces/msg/CommissioningState` |

Arm state/targets/services are reliable with bounded queues. Depth is
best-effort, keep-last-one at 15 Hz so delayed frames are dropped.

Foxy and Jazzy build the same tracked interface definitions. Unitree ROS 2 is
pinned and built separately for each distribution; generated artifacts are
never copied between machines.

## Network surface

| Host | Port | Purpose |
| --- | ---: | --- |
| GB10 | TCP 8000 | Browser UI, UI health/research, commissioning page |
| Robot → GB10 | UDP 5600 | H264 RTP RGB relay |
| Robot ↔ GB10 | dynamic UDP | ROS 2/CycloneDDS discovery and data |

CycloneDDS uses explicit peers, `rmw_cyclonedds_cpp`, and a shared
`ROS_DOMAIN_ID`; it does not depend on multicast discovery. The robot peer list
contains GB10 plus the Unitree control peer. There are no robot HTTP ports,
WebSockets, bearer tokens, or separate viewer on `8080`.

Keep the ROS network trusted. For a browser outside the LAN, forward only GB10
port `8000`. Protect an untrusted robot/GB10 network with a VPN or SROS2 rather
than recreating ad-hoc HTTP authentication.
