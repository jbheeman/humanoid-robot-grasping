#!/usr/bin/env python3
"""Headless, DDS-free Isaac replay of production G1 arm targets.

This intentionally drives the production :class:`ArmBridgeController` through
an in-process Isaac hardware adapter. It never imports Unitree DDS/ROS modules
and therefore cannot address the physical robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

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

if args.physics_hz < 100.0 or args.physics_hz > 250.0:
    parser.error("--physics-hz must be between 100 and 250")
for name in ("max_velocity", "max_acceleration", "max_jerk", "deadman_s", "tail_s"):
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be finite and positive")

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

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
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
)
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

    return getattr(value, "torch", value)


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


def main() -> int:
    assert_no_robot_transport_imports()
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
    robot.reset()
    robot.update(sim.get_physics_dt())

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
        write_joint_state(robot, joint_position, joint_velocity)
        robot.reset()

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
    bunny_cfg = RigidObjectCfg(
        prim_path="/World/BunnyProxy",
        spawn=sim_utils.SphereCfg(
            radius=0.055,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.82), roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(bunny_world.detach().cpu().tolist())),
    )
    bunny = RigidObject(bunny_cfg)
    sim.reset()
    robot.reset()
    bunny.reset()
    restore_body_state()

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
            weight_ramp_s=0.10,
            max_velocity_rad_s=args.max_velocity,
            max_acceleration_rad_s2=args.max_acceleration,
            max_jerk_rad_s3=args.max_jerk,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
    )
    calibration_id = str(replay.get("calibration_id") or "sim-calibration")
    controller.enable(session_id="isaac-replay", calibration_id=calibration_id)
    dt = 1.0 / args.physics_hz
    while controller.state is not ArmState.ARMED:
        controller.tick(clock.monotonic)
        hardware.apply()
        robot.write_data_to_sim()
        bunny.write_data_to_sim()
        sim.step(render=False)
        clock.advance(dt)
        robot.update(dt)
        bunny.update(dt)
        if clock.monotonic > 12.0:
            raise RuntimeError(f"sim controller did not arm: {controller.state_report()}")

    playback_start = clock.monotonic
    frame_index = 0
    sequence = 0
    final_time = float(frames[-1]["time_s"]) + args.tail_s
    minimum_hand_distance = float("inf")
    proximity_steps = 0
    tracking_errors: list[float] = []
    states: dict[str, int] = {}
    target_rejections: dict[str, int] = {}
    started_wall = time.perf_counter()
    initial_bunny = tensor(bunny.data.root_pos_w)[0].clone()
    while simulation_app.is_running() and clock.monotonic - playback_start <= final_time:
        elapsed = clock.monotonic - playback_start
        while frame_index < len(frames) and float(frames[frame_index]["time_s"]) <= elapsed:
            frame = frames[frame_index]
            if frame.get("status") == "target_sent" and isinstance(
                frame.get("right_arm_q_rad"), list
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
        controller.tick(clock.monotonic)
        report = controller.state_report(clock.monotonic)
        state = str(report["state"])
        states[state] = states.get(state, 0) + 1
        hardware.apply()
        robot.write_data_to_sim()
        bunny.write_data_to_sim()
        sim.step(render=False)
        clock.advance(dt)
        robot.update(dt)
        bunny.update(dt)
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

    final_bunny = tensor(bunny.data.root_pos_w)[0]
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
        "unitree_sim_commit": unitree_commit,
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
        "freshness_mode": args.freshness_mode,
        "actuator_profile": args.actuator_profile,
        "controller_state_steps": states,
        "target_rejections": target_rejections,
        "controller_final": controller.state_report(clock.monotonic),
        "minimum_hand_to_bunny_m": round(minimum_hand_distance, 5),
        "proximity_threshold_m": 0.09,
        "proximity_duration_s": round(proximity_steps * dt, 4),
        "physx_contact_measurement_available": False,
        "bunny_displacement_m": round(float(torch.linalg.vector_norm(final_bunny - initial_bunny)), 5),
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
    raise SystemExit(main())
finally:
    simulation_app.close()
