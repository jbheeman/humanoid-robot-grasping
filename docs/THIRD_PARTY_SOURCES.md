# Third-party robot sources

Setup fetches official Unitree sources into ignored `.deps/` directories and
builds generated artifacts under ignored `.ros/` directories. Preserve each
upstream license when redistributing source or assets.

| Project | Revision | Use |
| --- | --- | --- |
| [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) | `12c080cb91ee55854358ee9413c2e36e543c36ee` (v0.3.0) | `unitree_hg` and `unitree_api` messages; official ROS 2 G1 state, arm, motion-switcher, and locomotion contracts |
| [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6` | G1 29-DOF URDF/meshes and Pinocchio/CasADi IK reference |
| [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | `d96d8f63ae17a7108d4f7229c00ef875ba7129c9` | Cross-check of official G1 descriptions and joint metadata |

The former `unitree_sdk2_python`/Python CycloneDDS transport is no longer a
runtime dependency. Historical notes or ignored checkouts may still name it,
but production control uses generated ROS 2 messages through `rclpy`.

Repository-native code includes the `g1_control_interfaces` package, guarded
transport adapters, fusion/IK logic, and browser UI integration. Do not copy
upstream source or binary assets out of `.deps/` without their notices.
