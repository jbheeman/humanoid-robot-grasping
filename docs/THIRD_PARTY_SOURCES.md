# Third-Party Robot Sources

The arm pipeline uses official Unitree sources without copying or modifying their license text in this repository. `scripts/fetch_unitree_arm_assets.sh` checks out the exact revisions below under ignored `.deps/` directories and verifies the upstream license and required model files are present.

| Project | Revision | Use |
| --- | --- | --- |
| [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6` | G1 29-DOF URDF, meshes, 250 Hz `rt/arm_sdk` controller reference, and Pinocchio/CasADi IK reference |
| [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | `d96d8f63ae17a7108d4f7229c00ef875ba7129c9` | Cross-check of official G1 descriptions and joint metadata |
| [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | `37116c521f1588482e238d8450e471ba78ab9863` | Robot-local DDS and `rt/arm_sdk` transport |

The runtime wrapper in `src/object_tracking/arm_tracking/` is repository-native code. Do not copy upstream source or assets out of `.deps/` without also preserving the applicable upstream license and notices.
