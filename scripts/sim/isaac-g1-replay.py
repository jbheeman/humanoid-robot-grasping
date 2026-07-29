#!/usr/bin/env python3
"""Headless, DDS-free Isaac replay of production G1 arm targets.

This intentionally drives the production :class:`ArmBridgeController` through
an in-process Isaac hardware adapter. It never imports Unitree DDS/ROS modules
and therefore cannot address the physical robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("replay", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--project-root", type=Path, required=True)
parser.add_argument("--unitree-sim-root", type=Path, required=True)
parser.add_argument("--physics-hz", type=float, default=250.0)
parser.add_argument("--max-velocity", type=float, default=1.0)
parser.add_argument("--max-acceleration", type=float, default=4.0)
parser.add_argument("--max-jerk", type=float, default=30.0)
parser.add_argument("--deadman-s", type=float, default=0.75)
parser.add_argument("--tail-s", type=float, default=2.0)
parser.add_argument(
    "--command-source",
    choices=("recorded", "planned_approach", "closed_loop_ipc"),
    default="recorded",
    help="replay targets, run local IK, or stream measured state to the Linux planner",
)
parser.add_argument(
    "--planned-command-hz",
    type=float,
    default=30.0,
    help="measured-state IK/waypoint update rate for --command-source planned_approach",
)
parser.add_argument(
    "--planner-launcher",
    choices=("local", "wsl"),
    default="wsl" if os.name == "nt" else "local",
)
parser.add_argument(
    "--planner-project-root",
    default=None,
    help="planner-side POSIX project root required by --command-source closed_loop_ipc",
)
parser.add_argument("--planner-intercept-config", default=None)
parser.add_argument("--planner-wsl-distro", default=None)
parser.add_argument(
    "--planner-state-hz",
    type=float,
    default=30.0,
    help="newest-only measured-state update rate for the external planner",
)
parser.add_argument(
    "--planner-command-ttl-s",
    type=float,
    default=0.10,
    help="maximum simulated age of a planner response before it is ignored",
)
parser.add_argument(
    "--planner-max-realtime-factor",
    type=float,
    default=1.0,
    help="pace closed-loop simulation so wall-time planning remains meaningful",
)
parser.add_argument(
    "--bunny-motion",
    choices=("static", "recorded", "ballistic"),
    default="static",
    help="keep the proxy at rest, kinematically replay it, or launch it dynamically",
)
parser.add_argument(
    "--max-replay-s",
    type=float,
    default=None,
    help="optional smoke-test cap; full validation leaves this unset",
)
parser.add_argument(
    "--progress-interval-s",
    type=float,
    default=10.0,
    help="simulated seconds between progress records",
)
parser.add_argument(
    "--actuator-profile",
    choices=("unitree_official", "sdk_numeric"),
    default="unitree_official",
)
parser.add_argument(
    "--freshness-mode",
    choices=("recorded", "zero"),
    default="recorded",
    help="preserve production pipeline age for parity or force ideal freshness",
)
parser.add_argument(
    "--calibrated-frame",
    default="torso_link",
    help="exact USD body corresponding to the calibration torso frame",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# SimulationApp defaults to multi-GPU rendering when this key is omitted.
# Make the single-G1 validation process genuinely single-GPU before Kit starts;
# a late Kit setting still allows its IOMMU P2P probe to run on dual-GPU hosts.
args.multi_gpu = False
# The dual-3090 validation host has IOMMU enabled. Isaac's optional CUDA P2P
# diagnostic can stall before the stage opens even though one-GPU PhysX works.
# Disable only those startup diagnostics; physics and collision checking remain
# on the requested CUDA device.
required_kit_args = (
    "--/validate/iommu/enabled=false --/validate/p2p/enabled=false "
    "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
)
args.kit_args = f"{required_kit_args} {args.kit_args or ''}".strip()

if args.physics_hz < 100.0 or args.physics_hz > 250.0:
    parser.error("--physics-hz must be between 100 and 250")
for name in ("max_velocity", "max_acceleration", "max_jerk", "deadman_s", "tail_s"):
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be finite and positive")
if args.max_replay_s is not None and (
    not math.isfinite(args.max_replay_s) or args.max_replay_s <= 0.0
):
    parser.error("--max-replay-s must be finite and positive")
if not math.isfinite(args.progress_interval_s) or args.progress_interval_s <= 0.0:
    parser.error("--progress-interval-s must be finite and positive")
if (
    not math.isfinite(args.planned_command_hz)
    or args.planned_command_hz <= 0.0
    or args.planned_command_hz > args.physics_hz
):
    parser.error("--planned-command-hz must be finite, positive, and no faster than physics")
if args.command_source == "closed_loop_ipc" and (
    not args.planner_project_root or not args.planner_intercept_config
):
    parser.error(
        "--planner-project-root and --planner-intercept-config are required "
        "by --command-source closed_loop_ipc"
    )
if (
    not math.isfinite(args.planner_state_hz)
    or args.planner_state_hz <= 0.0
    or args.planner_state_hz > args.physics_hz
):
    parser.error("--planner-state-hz must be finite, positive, and no faster than physics")
if (
    not math.isfinite(args.planner_command_ttl_s)
    or args.planner_command_ttl_s <= 0.0
    or args.planner_command_ttl_s > args.deadman_s
):
    parser.error("--planner-command-ttl-s must be positive and no longer than deadman")
if (
    not math.isfinite(args.planner_max_realtime_factor)
    or args.planner_max_realtime_factor <= 0.0
    or args.planner_max_realtime_factor > 1.0
):
    parser.error("--planner-max-realtime-factor must be in (0, 1]")

# The official Unitree asset configuration reads PROJECT_ROOT at import time.
unitree_root = args.unitree_sim_root.resolve()
project_root = args.project_root.resolve()
if not (unitree_root / "robots/unitree.py").is_file():
    parser.error("--unitree-sim-root does not contain the pinned Unitree simulator")
if not (project_root / "src/object_tracking").is_dir():
    parser.error("--project-root does not contain this project")
os.environ["PROJECT_ROOT"] = str(unitree_root)
sys.path.insert(0, str(unitree_root))
sys.path.insert(0, str(project_root / "src"))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Lab 3's USD spawner imports POSIX ``fcntl`` even when its inter-process
# lock is disabled (LOCAL_WORLD_SIZE=1). Native Windows has no such module.
# Supply only the unused symbols so the single-process simulator can import;
# never emulate or silently weaken multi-process locking.
if os.name == "nt" and "fcntl" not in sys.modules:
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise RuntimeError("native Windows replay supports only LOCAL_WORLD_SIZE=1")
    fcntl_compat = types.ModuleType("fcntl")
    fcntl_compat.LOCK_EX = 2
    fcntl_compat.LOCK_UN = 8
    fcntl_compat.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = fcntl_compat

import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
if os.name == "nt":
    from isaaclab_physx.physics.physx_manager import PhysxManager  # noqa: E402
else:
    PhysxManager = None
from object_tracking.arm_tracking.arm_bridge import (  # noqa: E402
    ArmBridgeConfig,
    ArmBridgeController,
    ArmCommand,
    ArmState,
    RobotState,
)
from object_tracking.arm_tracking.joints import (  # noqa: E402
    BODY_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    joint_contract_id,
)
from object_tracking.arm_tracking.geometry import SupportRegion  # noqa: E402
from object_tracking.arm_tracking.gravity import UrdfGravityCompensator  # noqa: E402
from object_tracking.arm_tracking.ik_solver import (  # noqa: E402
    G1RightArmIK,
    default_urdf_path,
)
from object_tracking.arm_tracking.runtime import (  # noqa: E402
    compress_validated_joint_path,
    ruckig_edge_is_valid,
    select_start_escape_waypoint,
)
from object_tracking.arm_tracking.sim_closed_loop import (  # noqa: E402
    ObjectObservation,
    SimCommand,
    SimState,
)
from object_tracking.arm_tracking.sim_ipc import LatestPlannerProcess  # noqa: E402
from robots.unitree import G129_CFG_WITH_DEX1_BASE_FIX  # noqa: E402


EXPECTED_UNITREE_SIM_COMMIT = "e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc"


def assert_no_robot_transport_imports() -> None:
    forbidden = ("unitree_sdk2py", "cyclonedds", "rclpy", "dds")
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == token or name.startswith(f"{token}.") for token in forbidden)
    )
    if loaded:
        raise RuntimeError(f"simulation isolation failed; robot transport modules loaded: {loaded}")


def tensor(value):
    """Support Isaac Lab 2.x tensors and the 2026 typed tensor wrapper."""

    value = getattr(value, "torch", value)
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def find_indices(names: list[str], wanted: tuple[str, ...]) -> list[int]:
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise RuntimeError(f"official G1 asset is missing joints: {missing}")
    return [lookup[name] for name in wanted]


def quat_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    q = quaternion.reshape(1, 4)
    p = point.reshape(1, 3)
    return math_utils.quat_apply(q, p).reshape(3)


def local_to_world(
    torso_position: torch.Tensor,
    torso_quaternion: torch.Tensor,
    local_position,
) -> torch.Tensor:
    point = torch.as_tensor(local_position, dtype=torch.float32, device=torso_position.device)
    return torso_position + quat_apply(torso_quaternion, point)


def world_to_local(
    torso_position: torch.Tensor,
    torso_quaternion: torch.Tensor,
    world_position: torch.Tensor,
) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
        torso_quaternion.reshape(1, 4),
        (world_position - torso_position).reshape(1, 3),
    ).reshape(3)


def world_vector_to_local(
    torso_quaternion: torch.Tensor,
    world_vector: torch.Tensor,
) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
        torso_quaternion.reshape(1, 4),
        world_vector.reshape(1, 3),
    ).reshape(3)


def write_joint_state(robot: Articulation, position: torch.Tensor, velocity: torch.Tensor) -> None:
    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(position, velocity)
    else:
        robot.write_joint_position_to_sim_index(position=position)
        robot.write_joint_velocity_to_sim_index(velocity=velocity)


def set_position_target(
    robot: Articulation,
    target: torch.Tensor,
    joint_ids: list[int],
) -> None:
    try:
        robot.set_joint_position_target(target, joint_ids=joint_ids)
    except (AttributeError, TypeError):
        robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)


def set_velocity_target(
    robot: Articulation,
    target: torch.Tensor,
    joint_ids: list[int],
) -> None:
    try:
        robot.set_joint_velocity_target(target, joint_ids=joint_ids)
    except (AttributeError, TypeError):
        if hasattr(robot, "set_joint_velocity_target_index"):
            robot.set_joint_velocity_target_index(target=target, joint_ids=joint_ids)


def set_effort_target(
    robot: Articulation,
    target: torch.Tensor,
    joint_ids: list[int],
) -> None:
    try:
        robot.set_joint_effort_target(target, joint_ids=joint_ids)
    except (AttributeError, TypeError):
        if hasattr(robot, "set_joint_effort_target_index"):
            robot.set_joint_effort_target_index(target=target, joint_ids=joint_ids)


def write_root_pose(
    rigid_object: RigidObject,
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> None:
    pose = torch.cat((position.reshape(3), quaternion.reshape(4))).reshape(1, 7)
    rigid_object.write_root_pose_to_sim(pose)


def rebind_windows_physics_views(*assets) -> None:
    """Rebind Isaac Lab assets after the native-Windows PhysX reset."""

    if os.name != "nt":
        return
    assert PhysxManager is not None
    physics_manager = PhysxManager
    physics_manager._invalidate_views()
    physics_manager._warmup_needed = True
    physics_manager._warmup_and_create_views()
    for asset in assets:
        asset._invalidate_initialize_callback(None)
        asset._initialize_callback(None)


def prepare_windows_physics_reload() -> None:
    """Ensure newly spawned native-Windows bodies enter the next PhysX view."""

    if os.name != "nt":
        return
    assert PhysxManager is not None
    PhysxManager._invalidate_views()
    PhysxManager._warmup_needed = True


class IsaacClock:
    def __init__(self) -> None:
        self.monotonic = 10.0
        self.wall = 1_800_000_000.0

    def advance(self, dt: float) -> None:
        self.monotonic += dt
        self.wall += dt


class IsaacArmHardware:
    def __init__(
        self,
        robot: Articulation,
        left_ids: list[int],
        right_ids: list[int],
        clock: IsaacClock,
    ) -> None:
        self.robot = robot
        self.left_ids = left_ids
        self.right_ids = right_ids
        self.arm_ids = [*left_ids, *right_ids]
        self.clock = clock
        self.command: ArmCommand | None = None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def latest_state(self) -> RobotState:
        positions = tensor(self.robot.data.joint_pos)[0, self.arm_ids].detach().cpu().tolist()
        velocities = tensor(self.robot.data.joint_vel)[0, self.arm_ids].detach().cpu().tolist()
        return RobotState(
            arm_q=tuple(float(value) for value in positions),
            arm_dq=tuple(float(value) for value in velocities),
            received_at=self.clock.monotonic,
            standing=True,
            standing_since=self.clock.monotonic - 10.0,
            compatible_motion_mode=True,
            controller_available=True,
        )

    def publish(self, command: ArmCommand) -> None:
        self.command = command

    def apply(self) -> None:
        if self.command is None:
            return
        measured = tensor(self.robot.data.joint_pos)[:, self.arm_ids]
        device = measured.device
        requested_q = torch.tensor([self.command.q], dtype=torch.float32, device=device)
        requested_dq = torch.tensor([self.command.dq], dtype=torch.float32, device=device)
        requested_tau = torch.tensor([self.command.tau], dtype=torch.float32, device=device)
        # The official Isaac USD imports arm effort axes opposite to the
        # canonical URDF/SDK torque convention used by the production planner.
        requested_tau[:, 7:] *= -1.0
        # The real SDK's weight blends ownership away during HOLDING. Blending
        # the position target back to the current measured pose prevents the
        # fixed Isaac actuator stiffness from secretly holding a released arm.
        weight = self.command.weight
        q = measured + weight * (requested_q - measured)
        dq = weight * requested_dq
        tau = weight * requested_tau
        set_position_target(self.robot, q, self.arm_ids)
        set_velocity_target(self.robot, dq, self.arm_ids)
        set_effort_target(self.robot, tau, self.arm_ids)


def support_geometry(
    replay: dict,
) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    support = replay.get("support_plane")
    if not isinstance(support, dict):
        raise RuntimeError("replay is missing its calibrated support plane")
    footprint = support.get("footprint")
    if not isinstance(footprint, dict):
        raise RuntimeError("replay support plane is missing its bounded footprint")
    origin = footprint.get("origin")
    axis_u = footprint.get("axis_u")
    axis_v = footprint.get("axis_v")
    minimum = footprint.get("minimum_uv")
    maximum = footprint.get("maximum_uv")
    if not all(isinstance(value, list) for value in (origin, axis_u, axis_v, minimum, maximum)):
        raise RuntimeError("replay support footprint is incomplete")
    center = [
        origin[index]
        + 0.5 * (minimum[0] + maximum[0]) * axis_u[index]
        + 0.5 * (minimum[1] + maximum[1]) * axis_v[index]
        for index in range(3)
    ]
    normal = support.get("normal")
    if not isinstance(normal, list) or len(normal) != 3:
        raise RuntimeError("replay support plane normal is incomplete")
    return center, axis_u, axis_v, normal, maximum[0] - minimum[0], maximum[1] - minimum[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    assert_no_robot_transport_imports()
    project_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    project_tracked_dirty = bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
    )
    unitree_commit = subprocess.check_output(
        ["git", "-C", str(unitree_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if unitree_commit != EXPECTED_UNITREE_SIM_COMMIT:
        raise RuntimeError(
            f"Unitree simulator checkout {unitree_commit} is not pinned "
            f"{EXPECTED_UNITREE_SIM_COMMIT}"
        )
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay.get("schema_version") != 1:
        raise RuntimeError("unsupported replay schema")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("replay contains no frames")

    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / args.physics_hz,
            device=args.device,
            gravity=(0.0, 0.0, -9.81),
        )
    )
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    robot_cfg = G129_CFG_WITH_DEX1_BASE_FIX.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.spawn.activate_contact_sensors = True
    if args.actuator_profile == "sdk_numeric":
        # This is an experiment, not assumed parity: SDK gains and Isaac
        # implicit-drive gains share numbers but require step-response
        # calibration before their dynamics can be treated as equivalent.
        robot_cfg.actuators["arms"].stiffness = {
            ".*_shoulder_.*_joint": 80.0,
            ".*_elbow_joint": 80.0,
            ".*_wrist_.*_joint": 40.0,
        }
        robot_cfg.actuators["arms"].damping = {
            ".*_shoulder_.*_joint": 3.0,
            ".*_elbow_joint": 3.0,
            ".*_wrist_.*_joint": 1.5,
        }
    robot = Articulation(robot_cfg)
    sim.reset()
    rebind_windows_physics_views(robot)

    joint_names = list(robot.data.joint_names)
    left_ids = find_indices(joint_names, LEFT_ARM_JOINT_NAMES)
    right_ids = find_indices(joint_names, RIGHT_ARM_JOINT_NAMES)
    body_ids = find_indices(joint_names, BODY_JOINT_NAMES)
    initial_body = next(
        (
            frame.get("measured_body_q_rad")
            for frame in frames
            if isinstance(frame.get("measured_body_q_rad"), list)
        ),
        None,
    )
    initial_body_velocity = next(
        (
            frame.get("measured_body_dq_rad_s")
            for frame in frames
            if isinstance(frame.get("measured_body_dq_rad_s"), list)
        ),
        None,
    )
    if initial_body is None:
        raise RuntimeError("replay contains no complete measured 29-DOF initial state")

    planned_solver: G1RightArmIK | None = None
    planned_support: SupportRegion | None = None
    planned_path: tuple[tuple[float, ...], ...] | None = None
    planned_route: str | None = None
    planned_adaptive_reason: str | None = None
    planned_planning_ms: float | None = None
    planned_target_xyz: tuple[float, float, float] | None = None
    closed_loop_support: SupportRegion | None = None
    if args.command_source == "planned_approach":
        support_value = replay.get("support_plane")
        if not isinstance(support_value, dict):
            raise RuntimeError("planned approach requires a bounded support plane")
        planned_support = SupportRegion.from_dict(support_value)
        target_frame = next(
            (
                frame
                for frame in frames
                if frame.get("status") == "target_sent"
                and isinstance(frame.get("target_xyz_m"), list)
            ),
            None,
        )
        if target_frame is None:
            raise RuntimeError("planned approach requires a recorded Cartesian target")
        planned_target_xyz = tuple(float(value) for value in target_frame["target_xyz_m"])
        planned_solver = G1RightArmIK(default_urdf_path(project_root))
        initial_right = tuple(float(value) for value in initial_body[22:29])
        target_transform = planned_solver.forward_kinematics(initial_right)
        target_transform[:3, 3] = planned_target_xyz
        planning_started = time.perf_counter()
        planned = planned_solver.plan_adaptive_table_approach(
            target_transform,
            initial_right,
            support_plane=planned_support,
            top_clearance_m=0.11,
            final_validation_edge_step_rad=None,
        )
        planned_adaptive_reason = planned.reason
        planned_route = "adaptive"
        if not planned.ok or planned.q_path is None:
            planned = planned_solver.plan_guided_clearance(
                initial_right,
                support_plane=planned_support,
                lift_m=0.12,
                forward_m=0.04,
            )
            planned_route = "guided_fallback"
        if not planned.ok or planned.q_path is None:
            raise RuntimeError(f"production table approach failed: {planned.reason}")
        planned_path = compress_validated_joint_path(
            planned.q_path,
            lambda begin, end: ruckig_edge_is_valid(
                planned_solver,
                begin,
                end,
                support_plane=planned_support,
                maximum_velocity_rad_s=args.max_velocity,
                maximum_acceleration_rad_s2=args.max_acceleration,
                maximum_jerk_rad_s3=args.max_jerk,
            ),
            maximum_span_rad=0.70,
            maximum_skip_knots=32,
        )
        if planned_path is None:
            raise RuntimeError("production table approach compression failed")
        dense_error = planned_solver.validate_joint_path(
            planned_path,
            support_plane=planned_support,
            edge_step_rad=0.005,
            semantic_edge_step_rad=0.005,
            require_escape_cleared=False,
        )
        if dense_error is not None:
            raise RuntimeError(f"production table approach dense validation failed: {dense_error}")
        planned_planning_ms = (time.perf_counter() - planning_started) * 1000.0
    elif args.command_source == "closed_loop_ipc":
        support_value = replay.get("support_plane")
        if not isinstance(support_value, dict):
            raise RuntimeError("closed-loop planning requires a bounded support plane")
        closed_loop_support = SupportRegion.from_dict(support_value)

    def restore_body_state() -> None:
        joint_position = tensor(robot.data.default_joint_pos).clone()
        joint_velocity = tensor(robot.data.default_joint_vel).clone()
        joint_position[0, body_ids] = torch.tensor(
            initial_body,
            device=joint_position.device,
        )
        if initial_body_velocity is not None:
            joint_velocity[0, body_ids] = torch.tensor(
                initial_body_velocity,
                device=joint_velocity.device,
            )
        # Reset actuator/controller buffers before writing the captured state.
        # Resetting afterwards restores the USD default position targets even
        # though PhysX retains the written joint state.
        robot.reset()
        write_joint_state(robot, joint_position, joint_velocity)
        set_position_target(robot, joint_position[:, body_ids], body_ids)
        set_velocity_target(
            robot,
            torch.zeros_like(joint_velocity[:, body_ids]),
            body_ids,
        )

    # Compute the calibrated frame only after legs, waist, and both arms match
    # the captured physical state.
    restore_body_state()
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())
    body_names = list(robot.data.body_names)
    torso_candidates = [
        index for index, name in enumerate(body_names) if name == args.calibrated_frame
    ]
    hand_candidates = [
        index
        for index, name in enumerate(body_names)
        if name == "right_hand" or "right_hand" in name or name == "right_wrist_yaw_link"
    ]
    if not torso_candidates or not hand_candidates:
        raise RuntimeError(
            f"could not resolve calibrated frame {args.calibrated_frame!r} and right hand "
            f"from official asset: {body_names}"
        )
    torso_id = torso_candidates[0]
    hand_ids = hand_candidates
    torso_position = tensor(robot.data.body_pos_w)[0, torso_id]
    torso_quaternion = tensor(robot.data.body_quat_w)[0, torso_id]

    table_center, axis_u, axis_v, normal, table_x, table_y = support_geometry(replay)
    thickness = 0.035
    table_center = [
        coordinate - thickness / 2.0 * normal[index]
        for index, coordinate in enumerate(table_center)
    ]
    table_world = local_to_world(torso_position, torso_quaternion, table_center)
    local_basis = torch.tensor(
        [
            [axis_u[0], axis_v[0], normal[0]],
            [axis_u[1], axis_v[1], normal[1]],
            [axis_u[2], axis_v[2], normal[2]],
        ],
        dtype=torch.float32,
        device=torso_position.device,
    ).reshape(1, 3, 3)
    local_table_quaternion = math_utils.quat_from_matrix(local_basis).reshape(1, 4)
    table_quaternion = math_utils.quat_mul(
        torso_quaternion.reshape(1, 4),
        local_table_quaternion,
    ).reshape(4)
    table = sim_utils.CuboidCfg(
        size=(table_x, table_y, thickness),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.55, 0.65)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.05 if args.bunny_motion == "ballistic" else 0.8,
            dynamic_friction=0.02 if args.bunny_motion == "ballistic" else 0.7,
            restitution=0.05,
        ),
    )
    table.func(
        "/World/Table",
        table,
        translation=table_world.detach().cpu().tolist(),
        orientation=table_quaternion.detach().cpu().tolist(),
    )

    object_frames = [frame for frame in frames if isinstance(frame.get("object_xyz_m"), list)]
    if not object_frames:
        raise RuntimeError("replay contains no localized bunny position")
    bunny_world = local_to_world(
        torso_position,
        torso_quaternion,
        object_frames[0]["object_xyz_m"],
    )
    initial_bunny_velocity_local = torch.as_tensor(
        object_frames[0].get("object_velocity_m_s") or (0.0, 0.0, 0.0),
        dtype=torch.float32,
        device=torso_position.device,
    )
    initial_bunny_velocity_world = quat_apply(
        torso_quaternion,
        initial_bunny_velocity_local,
    )
    proxy = replay.get("object_proxy")
    if not isinstance(proxy, dict):
        proxy = {"shape": "sphere", "radius_m": 0.055}
    proxy_shape = str(proxy.get("shape") or "sphere")
    proxy_radius_m = float(proxy.get("radius_m") or 0.055)
    if proxy_shape == "capsule":
        bunny_spawn = sim_utils.CapsuleCfg(
            radius=proxy_radius_m,
            height=float(proxy.get("height_m") or 0.16),
            axis="Z",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=args.bunny_motion == "recorded",
                disable_gravity=args.bunny_motion == "ballistic",
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.82), roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0 if args.bunny_motion == "ballistic" else 0.8,
                dynamic_friction=0.0 if args.bunny_motion == "ballistic" else 0.7,
                restitution=0.05,
            ),
        )
    elif proxy_shape == "sphere":
        bunny_spawn = sim_utils.SphereCfg(
            radius=proxy_radius_m,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=args.bunny_motion == "recorded",
                disable_gravity=args.bunny_motion == "ballistic",
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.82), roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0 if args.bunny_motion == "ballistic" else 0.8,
                dynamic_friction=0.0 if args.bunny_motion == "ballistic" else 0.7,
                restitution=0.05,
            ),
        )
    else:
        raise RuntimeError(f"unsupported object proxy shape: {proxy_shape}")
    bunny_cfg = RigidObjectCfg(
        prim_path="/World/BunnyProxy",
        spawn=bunny_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(bunny_world.detach().cpu().tolist()),
            lin_vel=(
                tuple(initial_bunny_velocity_world.detach().cpu().tolist())
                if args.bunny_motion == "ballistic"
                else (0.0, 0.0, 0.0)
            ),
        ),
    )
    bunny = RigidObject(bunny_cfg)
    contact_sensors = [
        ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/G1/{body_names[index]}",
                update_period=0.0,
                history_length=1,
                filter_prim_paths_expr=["/World/BunnyProxy"],
            )
        )
        for index in hand_candidates
    ]
    prepare_windows_physics_reload()
    sim.reset()
    rebind_windows_physics_views(robot, bunny, *contact_sensors)
    robot.reset()
    bunny.reset()
    for sensor in contact_sensors:
        sensor.reset()
    restore_body_state()
    # ``write_joint_state_to_sim`` updates PhysX immediately, but after the
    # second stage reset Isaac Lab's cached ``robot.data`` can still contain
    # the USD default pose.  ArmBridge.enable() must latch the replay's
    # measured pose, not that stale default, or the zero-weight arming command
    # pulls the arm away before the startup settle check can complete.
    robot.write_data_to_sim()
    robot.update(0.0)

    clock = IsaacClock()
    hardware = IsaacArmHardware(robot, left_ids, right_ids, clock)
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            allow_movement=True,
            calibration_id=str(replay.get("calibration_id") or "sim-calibration"),
            control_hz=args.physics_hz,
            target_ttl_s=0.5,
            deadman_s=args.deadman_s,
            stable_standing_s=0.01,
            startup_settle_s=0.02,
            startup_settle_timeout_s=1.0,
            # Native PhysX starts from a passive USD pose, unlike the real G1
            # whose low-level controller is already holding the measured pose.
            # Permit that one-time model transient; normal following-error,
            # collision, TTL, and deadman checks remain active after arming.
            startup_max_velocity_rad_s=0.20 if os.name == "nt" else 0.08,
            startup_max_pose_error_rad=0.20 if os.name == "nt" else 0.02,
            weight_ramp_s=0.10,
            max_velocity_rad_s=args.max_velocity,
            max_acceleration_rad_s2=args.max_acceleration,
            max_jerk_rad_s3=args.max_jerk,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
    )
    calibration_id = str(replay.get("calibration_id") or "sim-calibration")
    planner_client: LatestPlannerProcess | None = None
    planner_state_sequence = 0
    warmup_command: SimCommand | None = None
    warmup_observation_monotonic = clock.monotonic
    if args.command_source == "closed_loop_ipc":
        assert closed_loop_support is not None
        planner_root = str(args.planner_project_root)
        planner_command = [
            "uv",
            "run",
            "python",
            "scripts/sim/g1-closed-loop-planner.py",
            "--project-root",
            planner_root,
            "--intercept-config",
            str(args.planner_intercept_config),
            "--max-velocity",
            str(args.max_velocity),
            "--max-acceleration",
            str(args.max_acceleration),
            "--max-jerk",
            str(args.max_jerk),
        ]
        if args.planner_launcher == "wsl":
            planner_command = [
                "wsl.exe",
                *(
                    []
                    if args.planner_wsl_distro is None
                    else ["--distribution", args.planner_wsl_distro]
                ),
                "--cd",
                planner_root,
                "--",
                *planner_command,
            ]
            planner_cwd = None
        else:
            planner_cwd = planner_root
        planner_client = LatestPlannerProcess(planner_command, cwd=planner_cwd)
        planner_client.start()
        first_object_frame = object_frames[0]
        first_object_position = tuple(
            float(value) for value in first_object_frame["object_xyz_m"]
        )
        first_object_velocity = tuple(
            float(value)
            for value in (
                first_object_frame.get("object_velocity_m_s")
                if isinstance(first_object_frame.get("object_velocity_m_s"), list)
                else (0.0, 0.0, 0.0)
            )
        )
        warmup_observation = ObjectObservation(
            track_id=int(first_object_frame.get("track_id") or 1),
            class_name=str(first_object_frame.get("class_name") or "bunny"),
            confidence=float(first_object_frame.get("confidence") or 1.0),
            position_m=first_object_position,
            velocity_m_s=first_object_velocity,
            # Planner protocol time is episode-relative. IsaacClock carries a
            # nonzero monotonic offset for controller freshness tests.
            observation_time_s=0.0,
            consecutive_observations=int(
                first_object_frame.get("estimator_consecutive_observations") or 2
            ),
            residual_m=float(first_object_frame.get("estimator_residual_m") or 0.0),
        )
        measured_body_q = tuple(
            float(value)
            for value in tensor(robot.data.joint_pos)[0, body_ids].detach().cpu().tolist()
        )
        measured_body_dq = tuple(
            float(value)
            for value in tensor(robot.data.joint_vel)[0, body_ids].detach().cpu().tolist()
        )
        warmup_command = planner_client.submit_and_wait(
            SimState(
                episode_id=f"isaac-{args.replay.stem}",
                sequence=planner_state_sequence,
                simulation_time_s=0.0,
                calibration_id=calibration_id,
                joint_contract_id=joint_contract_id(),
                body_q_rad=measured_body_q,
                body_dq_rad_s=measured_body_dq,
                support_region=closed_loop_support,
                object_observation=warmup_observation,
            ),
            timeout_s=30.0,
        )
        if warmup_command.status != "target":
            raise RuntimeError(
                "planner startup did not produce a target: "
                f"{warmup_command.status}:{warmup_command.reason}"
            )
        print(
            json.dumps(
                {
                    "event": "isaac_planner_warmup_ready",
                    "planning_latency_ms": warmup_command.planning_latency_ms,
                    "reason": warmup_command.reason,
                    "state_sequence": warmup_command.state_sequence,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    controller.enable(session_id="isaac-replay", calibration_id=calibration_id)
    dt = 1.0 / args.physics_hz
    startup_arm_position = torch.tensor(
        [[*initial_body[15:22], *initial_body[22:29]]],
        dtype=torch.float32,
        device=tensor(robot.data.joint_pos).device,
    )
    startup_arm_velocity = torch.zeros_like(startup_arm_position)
    startup_right_tau = UrdfGravityCompensator(
        default_urdf_path(project_root)
    ).torque(initial_body[22:29])
    startup_arm_effort = torch.tensor(
        [[0.0] * 7 + list(startup_right_tau)],
        dtype=torch.float32,
        device=startup_arm_position.device,
    )
    while controller.state is not ArmState.ARMED:
        controller.tick(clock.monotonic)
        hardware.apply()
        if controller.state is ArmState.ARMING:
            # Weight zero represents the handoff from the G1's underlying
            # standing controller, which already holds the measured pose.
            # Isaac has no such controller, so preserve only that startup hold
            # until ArmBridge completes its measured-pose takeover.
            set_position_target(robot, startup_arm_position, hardware.arm_ids)
            set_velocity_target(robot, startup_arm_velocity, hardware.arm_ids)
            set_effort_target(robot, startup_arm_effort, hardware.arm_ids)
        robot.write_data_to_sim()
        bunny.write_data_to_sim()
        sim.step(render=False)
        clock.advance(dt)
        robot.update(dt)
        bunny.update(dt)
        for sensor in contact_sensors:
            sensor.update(dt)
        if clock.monotonic > 12.0:
            raise RuntimeError(f"sim controller did not arm: {controller.state_report()}")

    sequence = 0
    if warmup_command is not None:
        sequence += 1
        controller.set_target(
            session_id="isaac-replay",
            sequence=sequence,
            calibration_id=calibration_id,
            right_arm_q=warmup_command.right_arm_q_rad,
            right_arm_tau_ff=warmup_command.right_arm_tau_ff_nm,
            pipeline_age_ms=int(
                round(
                    max(
                        0.0,
                        clock.monotonic - warmup_observation_monotonic,
                    )
                    * 1000.0
                )
            ),
        )
    if args.bunny_motion == "ballistic":
        # Controller takeover advances physics before episode time starts.
        # Relaunch the object here so the planner observes the scenario's
        # intended crossing velocity instead of a plush already slowed by its
        # initial table contact.
        write_root_pose(
            bunny,
            bunny_world,
            tensor(bunny.data.root_quat_w)[0].clone(),
        )
        bunny.write_root_velocity_to_sim(
            torch.cat(
                (
                    initial_bunny_velocity_world.reshape(3),
                    torch.zeros(
                        3,
                        dtype=initial_bunny_velocity_world.dtype,
                        device=initial_bunny_velocity_world.device,
                    ),
                )
            ).reshape(1, 6)
        )
        bunny.update(0.0)

    playback_start = clock.monotonic
    frame_index = 0
    full_replay_time = float(frames[-1]["time_s"]) + args.tail_s
    final_time = (
        full_replay_time if args.max_replay_s is None else min(full_replay_time, args.max_replay_s)
    )
    minimum_hand_distance = float("inf")
    proximity_steps = 0
    tracking_errors: list[float] = []
    states: dict[str, int] = {}
    target_rejections: dict[str, int] = {}
    started_wall = time.perf_counter()
    next_progress_at = 0.0
    initial_bunny = tensor(bunny.data.root_pos_w)[0].clone()
    initial_bunny_velocity = tensor(bunny.data.root_lin_vel_w)[0].clone()
    bunny_orientation = tensor(bunny.data.root_quat_w)[0].clone()
    planned_target_index = 1
    planned_approach_complete_at_s: float | None = None
    planned_next_command_at_s = 0.0
    planned_commands = 0
    planned_edge_validations = 0
    planned_ik_failures: dict[str, int] = {}
    planner_last_applied_sequence = (
        -1 if warmup_command is None else warmup_command.state_sequence
    )
    planner_next_state_at_s = 0.0
    planner_target_commands = int(warmup_command is not None)
    planner_expired_commands = 0
    planner_response_statuses: dict[str, int] = {}
    current_bunny_local = tuple(float(value) for value in object_frames[0]["object_xyz_m"])
    current_bunny_velocity = tuple(
        float(value)
        for value in (
            object_frames[0].get("object_velocity_m_s")
            if isinstance(object_frames[0].get("object_velocity_m_s"), list)
            else (0.0, 0.0, 0.0)
        )
    )
    pending_object_observation: ObjectObservation | None = None
    contact_steps = 0
    maximum_contact_force_n = 0.0
    first_contact_at_s: float | None = None
    last_contact_at_s: float | None = None
    contact_measurement_available = False
    while simulation_app.is_running() and clock.monotonic - playback_start <= final_time:
        elapsed = clock.monotonic - playback_start
        if args.command_source == "closed_loop_ipc":
            minimum_wall_s = elapsed / args.planner_max_realtime_factor
            ahead_s = minimum_wall_s - (time.perf_counter() - started_wall)
            if ahead_s > 0.0:
                time.sleep(ahead_s)
        if elapsed >= next_progress_at:
            print(
                json.dumps(
                    {
                        "event": "isaac_replay_progress",
                        "elapsed_s": round(elapsed, 3),
                        "final_time_s": round(final_time, 3),
                        "controller_state": controller.state.value,
                        "frame_index": frame_index,
                        "target_sequence": sequence,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_progress_at += args.progress_interval_s
        while frame_index < len(frames) and float(frames[frame_index]["time_s"]) <= elapsed:
            frame = frames[frame_index]
            if frame.get("status") == "target_sent" and isinstance(frame.get("target_xyz_m"), list):
                planned_target_xyz = tuple(float(value) for value in frame["target_xyz_m"])
            if isinstance(frame.get("object_xyz_m"), list):
                current_bunny_local = tuple(float(value) for value in frame["object_xyz_m"])
            if isinstance(frame.get("object_velocity_m_s"), list):
                current_bunny_velocity = tuple(
                    float(value) for value in frame["object_velocity_m_s"]
                )
            if args.bunny_motion == "ballistic" and contact_steps > 0:
                current_bunny_local = tuple(
                    float(value)
                    for value in world_to_local(
                        torso_position,
                        torso_quaternion,
                        tensor(bunny.data.root_pos_w)[0],
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                current_bunny_velocity = tuple(
                    float(value)
                    for value in world_vector_to_local(
                        torso_quaternion,
                        tensor(bunny.data.root_lin_vel_w)[0],
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
            if isinstance(frame.get("object_xyz_m"), list):
                pending_object_observation = ObjectObservation(
                    track_id=int(frame.get("track_id") or 1),
                    class_name=str(frame.get("class_name") or "bunny"),
                    confidence=float(frame.get("confidence") or 1.0),
                    position_m=current_bunny_local,
                    velocity_m_s=current_bunny_velocity,
                    observation_time_s=float(frame["time_s"]),
                    consecutive_observations=int(
                        frame.get("estimator_consecutive_observations") or frame_index + 1
                    ),
                    residual_m=float(frame.get("estimator_residual_m") or 0.0),
                )
            if args.bunny_motion == "recorded" and isinstance(frame.get("object_xyz_m"), list):
                bunny_local_position = torch.as_tensor(
                    frame["object_xyz_m"],
                    dtype=torch.float32,
                    device=torso_position.device,
                )
                write_root_pose(
                    bunny,
                    local_to_world(
                        torso_position,
                        torso_quaternion,
                        bunny_local_position,
                    ),
                    bunny_orientation,
                )
            if (
                args.command_source == "recorded"
                and frame.get("status") == "target_sent"
                and isinstance(frame.get("right_arm_q_rad"), list)
            ):
                sequence += 1
                try:
                    controller.set_target(
                        session_id="isaac-replay",
                        sequence=sequence,
                        calibration_id=calibration_id,
                        right_arm_q=frame["right_arm_q_rad"],
                        right_arm_tau_ff=frame.get("right_arm_tau_ff_nm") or [0.0] * 7,
                        pipeline_age_ms=(
                            0
                            if args.freshness_mode == "zero"
                            else int(round(float(frame.get("pipeline_age_ms") or 0.0)))
                        ),
                    )
                except ValueError as exc:
                    code = str(getattr(exc, "code", type(exc).__name__))
                    target_rejections[code] = target_rejections.get(code, 0) + 1
            frame_index += 1
        if (
            planner_client is not None
            and closed_loop_support is not None
            and elapsed >= planner_next_state_at_s
        ):
            planner_next_state_at_s = elapsed + 1.0 / args.planner_state_hz
            measured_body_q = tuple(
                float(value)
                for value in tensor(robot.data.joint_pos)[0, body_ids].detach().cpu().tolist()
            )
            measured_body_dq = tuple(
                float(value)
                for value in tensor(robot.data.joint_vel)[0, body_ids].detach().cpu().tolist()
            )
            planner_state_sequence += 1
            latest = planner_client.submit(
                SimState(
                    episode_id=f"isaac-{args.replay.stem}",
                    sequence=planner_state_sequence,
                    simulation_time_s=elapsed,
                    calibration_id=calibration_id,
                    joint_contract_id=joint_contract_id(),
                    body_q_rad=measured_body_q,
                    body_dq_rad_s=measured_body_dq,
                    support_region=closed_loop_support,
                    object_observation=pending_object_observation,
                )
            )
            pending_object_observation = None
            if (
                latest is not None
                and latest.episode_id == f"isaac-{args.replay.stem}"
                and latest.state_sequence > planner_last_applied_sequence
                and elapsed - latest.simulation_time_s <= args.planner_command_ttl_s
            ):
                planner_last_applied_sequence = latest.state_sequence
                planner_response_statuses[latest.status] = (
                    planner_response_statuses.get(latest.status, 0) + 1
                )
                if latest.status == "target":
                    sequence += 1
                    try:
                        controller.set_target(
                            session_id="isaac-replay",
                            sequence=sequence,
                            calibration_id=calibration_id,
                            right_arm_q=latest.right_arm_q_rad,
                            right_arm_tau_ff=latest.right_arm_tau_ff_nm,
                            pipeline_age_ms=int(
                                round(
                                    max(
                                        0.0,
                                        elapsed
                                        - (
                                            elapsed
                                            if latest.source_observation_time_s is None
                                            else latest.source_observation_time_s
                                        ),
                                    )
                                    * 1000.0
                                )
                            ),
                        )
                        planner_target_commands += 1
                    except ValueError as exc:
                        code = str(getattr(exc, "code", type(exc).__name__))
                        target_rejections[code] = target_rejections.get(code, 0) + 1
            elif (
                latest is not None
                and latest.state_sequence > planner_last_applied_sequence
            ):
                planner_last_applied_sequence = latest.state_sequence
                planner_expired_commands += 1
        if args.command_source == "planned_approach" and elapsed >= planned_next_command_at_s:
            assert planned_solver is not None
            assert planned_support is not None
            assert planned_path is not None
            planned_next_command_at_s = elapsed + 1.0 / args.planned_command_hz
            measured = tuple(
                float(value)
                for value in tensor(robot.data.joint_pos)[0, right_ids].detach().cpu().tolist()
            )
            measured_velocity = tuple(
                float(value)
                for value in tensor(robot.data.joint_vel)[0, right_ids].detach().cpu().tolist()
            )
            desired_q: tuple[float, ...] | None = None
            if planned_approach_complete_at_s is None:
                desired_q, planned_target_index, selection_error = select_start_escape_waypoint(
                    measured,
                    planned_path,
                    planned_target_index,
                    reached_tolerance_rad=0.004,
                    measured_velocity_rad_s=measured_velocity,
                )
                if selection_error is not None:
                    raise RuntimeError(f"planned approach tracking failed: {selection_error}")
                if desired_q is None:
                    planned_approach_complete_at_s = elapsed
            if planned_approach_complete_at_s is not None and planned_target_xyz is not None:
                transform = planned_solver.forward_kinematics(measured)
                transform[:3, 3] = planned_target_xyz
                local = planned_solver.solve_local_translation(
                    transform,
                    measured,
                    support_plane=planned_support,
                )
                if local.ok and local.q_rad is not None:
                    desired_q = local.q_rad
                else:
                    reason = str(local.reason or "unknown")
                    planned_ik_failures[reason] = planned_ik_failures.get(reason, 0) + 1
                    desired_q = None
            if desired_q is not None:
                planned_edge_validations += 1
                if not ruckig_edge_is_valid(
                    planned_solver,
                    measured,
                    desired_q,
                    support_plane=planned_support,
                    maximum_velocity_rad_s=args.max_velocity,
                    maximum_acceleration_rad_s2=args.max_acceleration,
                    maximum_jerk_rad_s3=args.max_jerk,
                    current_velocity_rad_s=measured_velocity,
                ):
                    raise RuntimeError("planned measured Ruckig edge failed")
                sequence += 1
                try:
                    controller.set_target(
                        session_id="isaac-replay",
                        sequence=sequence,
                        calibration_id=calibration_id,
                        right_arm_q=desired_q,
                        right_arm_tau_ff=planned_solver.gravity_compensation_torque(desired_q),
                        pipeline_age_ms=0,
                    )
                    planned_commands += 1
                except ValueError as exc:
                    code = str(getattr(exc, "code", type(exc).__name__))
                    target_rejections[code] = target_rejections.get(code, 0) + 1
        controller.tick(clock.monotonic)
        report = controller.state_report(clock.monotonic)
        state = str(report["state"])
        states[state] = states.get(state, 0) + 1
        hardware.apply()
        if args.bunny_motion == "ballistic" and contact_steps == 0:
            # Model an externally driven plush/conveyor until interception.
            # Release it immediately after the first measured impact so the
            # post-contact velocity change remains a real PhysX outcome.
            write_root_pose(
                bunny,
                bunny_world + elapsed * initial_bunny_velocity_world,
                bunny_orientation,
            )
            bunny.write_root_velocity_to_sim(
                torch.cat(
                    (
                        initial_bunny_velocity_world.reshape(3),
                        torch.zeros(
                            3,
                            dtype=initial_bunny_velocity_world.dtype,
                            device=initial_bunny_velocity_world.device,
                        ),
                    )
                ).reshape(1, 6)
            )
        robot.write_data_to_sim()
        bunny.write_data_to_sim()
        sim.step(render=False)
        clock.advance(dt)
        robot.update(dt)
        bunny.update(dt)
        contact_force_n = 0.0
        for sensor in contact_sensors:
            sensor.update(dt)
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is not None:
                contact_measurement_available = True
                contact_force_n = max(
                    contact_force_n,
                    float(torch.linalg.vector_norm(tensor(force_matrix), dim=-1).max()),
                )
        maximum_contact_force_n = max(maximum_contact_force_n, contact_force_n)
        if contact_force_n > 1.0:
            contact_steps += 1
            if first_contact_at_s is None:
                first_contact_at_s = elapsed
            last_contact_at_s = elapsed
        hand_positions = tensor(robot.data.body_pos_w)[0, hand_ids]
        bunny_position = tensor(bunny.data.root_pos_w)[0]
        distance = float(torch.linalg.vector_norm(hand_positions - bunny_position, dim=1).min())
        minimum_hand_distance = min(minimum_hand_distance, distance)
        if distance <= 0.09:
            proximity_steps += 1
        if hardware.command is not None:
            measured = tensor(robot.data.joint_pos)[0, right_ids]
            commanded = torch.tensor(
                hardware.command.q[7:],
                dtype=measured.dtype,
                device=measured.device,
            )
            tracking_errors.append(float(torch.max(torch.abs(measured - commanded))))

    planner_metrics = None if planner_client is None else planner_client.metrics()
    if planner_client is not None:
        planner_client.close()
    final_bunny = tensor(bunny.data.root_pos_w)[0]
    final_bunny_velocity = tensor(bunny.data.root_lin_vel_w)[0]
    initial_bunny_velocity_local = world_vector_to_local(
        torso_quaternion,
        initial_bunny_velocity,
    )
    final_bunny_velocity_local = world_vector_to_local(
        torso_quaternion,
        final_bunny_velocity,
    )
    wall_elapsed = time.perf_counter() - started_wall
    ordered_errors = sorted(tracking_errors)
    p95_error = (
        None
        if not ordered_errors
        else ordered_errors[min(len(ordered_errors) - 1, round(0.95 * (len(ordered_errors) - 1)))]
    )
    result = {
        "schema_version": 1,
        "replay": str(args.replay),
        "project_commit": project_commit,
        "project_tracked_dirty": project_tracked_dirty,
        "unitree_sim_commit": unitree_commit,
        "replay_sha256": sha256_file(args.replay),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "planner_intercept_config_sha256": (
            None
            if args.planner_intercept_config is None
            or not Path(args.planner_intercept_config).is_file()
            else sha256_file(Path(args.planner_intercept_config))
        ),
        "planner_urdf_sha256": sha256_file(default_urdf_path(project_root)),
        "dds_enabled": False,
        "ros_enabled": False,
        "transport_import_guard_passed": True,
        "physics_hz": args.physics_hz,
        "limits": {
            "velocity_rad_s": args.max_velocity,
            "acceleration_rad_s2": args.max_acceleration,
            "jerk_rad_s3": args.max_jerk,
            "deadman_s": args.deadman_s,
        },
        "joint_mapping": {
            "body": dict(zip(BODY_JOINT_NAMES, find_indices(joint_names, BODY_JOINT_NAMES))),
            "left": dict(zip(LEFT_ARM_JOINT_NAMES, left_ids)),
            "right": dict(zip(RIGHT_ARM_JOINT_NAMES, right_ids)),
        },
        "calibrated_frame": args.calibrated_frame,
        "command_source": args.command_source,
        "bunny_motion": args.bunny_motion,
        "object_proxy": proxy,
        "freshness_mode": args.freshness_mode,
        "actuator_profile": args.actuator_profile,
        "planned_approach": (
            None
            if args.command_source != "planned_approach"
            else {
                "route": planned_route,
                "adaptive_reason": planned_adaptive_reason,
                "planning_ms": (
                    None if planned_planning_ms is None else round(planned_planning_ms, 3)
                ),
                "waypoints": None if planned_path is None else len(planned_path),
                "command_hz": args.planned_command_hz,
                "commands": planned_commands,
                "measured_edge_validations": planned_edge_validations,
                "approach_complete_at_s": (
                    None
                    if planned_approach_complete_at_s is None
                    else round(planned_approach_complete_at_s, 4)
                ),
                "local_ik_failures": planned_ik_failures,
            }
        ),
        "closed_loop_ipc": (
            None
            if planner_metrics is None
            else {
                "planner_launcher": args.planner_launcher,
                "planner_project_root": args.planner_project_root,
                "state_hz": args.planner_state_hz,
                "command_ttl_s": args.planner_command_ttl_s,
                "maximum_realtime_factor": args.planner_max_realtime_factor,
                "states_submitted": planner_metrics.states_submitted,
                "pending_states_replaced": planner_metrics.pending_states_replaced,
                "commands_received": planner_metrics.commands_received,
                "stale_commands": planner_metrics.stale_commands,
                "process_starts": planner_metrics.process_starts,
                "last_error": planner_metrics.last_error,
                "target_commands": planner_target_commands,
                "expired_commands": planner_expired_commands,
                "response_statuses": planner_response_statuses,
            }
        ),
        "controller_state_steps": states,
        "target_rejections": target_rejections,
        "controller_final": controller.state_report(clock.monotonic),
        "minimum_hand_to_bunny_m": round(minimum_hand_distance, 5),
        "proximity_threshold_m": 0.09,
        "proximity_duration_s": round(proximity_steps * dt, 4),
        "physx_contact_measurement_available": contact_measurement_available,
        "contact": {
            "threshold_n": 1.0,
            "maximum_force_n": round(maximum_contact_force_n, 5),
            "duration_s": round(contact_steps * dt, 4),
            "first_at_s": None if first_contact_at_s is None else round(first_contact_at_s, 4),
            "last_at_s": None if last_contact_at_s is None else round(last_contact_at_s, 4),
        },
        "bunny_displacement_m": round(
            float(torch.linalg.vector_norm(final_bunny - initial_bunny)), 5
        ),
        "bunny_initial_speed_m_s": round(
            float(torch.linalg.vector_norm(initial_bunny_velocity)), 5
        ),
        "bunny_initial_velocity_local_m_s": [
            round(float(value), 6) for value in initial_bunny_velocity_local
        ],
        "bunny_final_speed_m_s": round(
            float(torch.linalg.vector_norm(final_bunny_velocity)), 5
        ),
        "bunny_final_velocity_local_m_s": [
            round(float(value), 6) for value in final_bunny_velocity_local
        ],
        "bunny_velocity_change_m_s": round(
            float(torch.linalg.vector_norm(final_bunny_velocity - initial_bunny_velocity)),
            5,
        ),
        "joint_tracking_error_rad_p95": None if p95_error is None else round(p95_error, 6),
        "simulated_duration_s": round(final_time, 4),
        "wall_duration_s": round(wall_elapsed, 4),
        "realtime_factor": round(final_time / max(wall_elapsed, 1e-9), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


try:
    exit_code = main()
except BaseException as exc:
    print(
        json.dumps(
            {
                "event": "isaac_replay_fatal",
                "exception": type(exc).__name__,
                "message": str(exc),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    raise
else:
    raise SystemExit(exit_code)
finally:
    simulation_app.close()
