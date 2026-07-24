#!/usr/bin/env python3
"""Generate validated latency-aware moving-plush interception demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-PickPlace-RedBlock-G129-Dex1-Joint")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--seed-start", type=int, default=10000)
parser.add_argument("--bunny-usd", type=Path, required=True)
parser.add_argument("--table-usd", type=Path, required=True)
parser.add_argument("--robot-usd", type=Path, required=True)
parser.add_argument("--robot-urdf", type=Path, required=True)
parser.add_argument("--max-attempt-factor", type=float, default=2.0)
parser.add_argument("--min-free-gb", type=float, default=220.0)
parser.add_argument("--keep-rejected", type=int, default=5)
parser.add_argument("--min-initial-separation", type=float, default=0.08)
parser.add_argument("--max-precontact-force", type=float, default=0.75)
parser.add_argument("--max-raise-displacement", type=float, default=0.002)
parser.add_argument("--contact-force-threshold", type=float, default=0.5)
parser.add_argument("--pre-capture-tracking-tolerance", type=float, default=0.03)
parser.add_argument("--minimum-reaction-s", type=float, default=0.50)
parser.add_argument("--minimum-stop-fraction", type=float, default=0.70)
parser.add_argument("--maximum-postcontact-speed", type=float, default=0.035)
parser.add_argument("--maximum-right-table-force", type=float, default=0.50)
parser.add_argument("--maximum-initial-spin-rad-s", type=float, default=0.10)
parser.add_argument("--diagnostic-design-index", type=int)
parser.add_argument(
    "--rollout-output",
    type=Path,
    help=(
        "write one unbiased oracle_ik record per attempted seed; --episodes "
        "then means exact attempts rather than accepted replacements"
    ),
)
parser.add_argument(
    "--camera-variant",
    choices=("head", "angled-left", "angled-right"),
    default="head",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

root = Path(os.environ["G1_BUNNY_PROJECT_ROOT"])
sim_root = Path(os.environ["UNITREE_SIM_ROOT"])
sys.path[:0] = [str(root / "src"), str(sim_root)]
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402
import tasks  # noqa: E402,F401
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from g1_bunny_vla.contract import FrameSample  # noqa: E402
from g1_bunny_vla.episode_writer import EpisodeWriter  # noqa: E402
from g1_bunny_vla.validate_dataset import validate_episode  # noqa: E402

LEFT_ARM = ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")
RIGHT_ARM = tuple(name.replace("left_", "right_") for name in LEFT_ARM)
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
REAL_LEFT_HOLD = np.array(
    # Unitree's official G1 whole-body neutral arm-at-side posture.  Keeping
    # this physical (rather than hiding geometry) preserves collision truth
    # while placing the palm below and beside the tabletop/head-camera view.
    [0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
INSTRUCTION = "Block the moving rabbit plush with the right palm."
# The task's stock 6 cm cube is initialized at z=0.84, so its bottom
# establishes the actual tabletop surface at 0.810 m.
TABLE_Z = 0.810
RIGHT_BRAINCO_CONTACT_EXPR = (
    "right_(base_link|thumb_.*_link|thumb_tip|index_.*_link|index_tip|"
    "middle_.*_link|middle_tip|ring_.*_link|ring_tip|pinky_.*_link|pinky_tip)"
)
LEFT_BRAINCO_CONTACT_EXPR = RIGHT_BRAINCO_CONTACT_EXPR.replace("right_", "left_", 1)
SPEED_BINS_M_S = ((0.08, 0.10), (0.11, 0.13), (0.14, 0.16))
HEADING_BINS_DEG = ((-3.0, -1.0), (-0.5, 0.5), (1.0, 3.0))
TUCK_TARGETS = (
    (0.170, -0.185, 0.175),
    (0.185, -0.165, 0.200),
    (0.175, -0.200, 0.210),
)


def design_parameters(seed, rng):
    """Deterministic stratification; random seeds fill each bin without cloning it."""
    index = args.diagnostic_design_index
    if index is None:
        index = seed - args.seed_start
    speed_low, speed_high = SPEED_BINS_M_S[index % len(SPEED_BINS_M_S)]
    heading_low, heading_high = HEADING_BINS_DEG[(index // 3) % len(HEADING_BINS_DEG)]
    speed = rng.uniform(speed_low, speed_high)
    heading_deg = rng.uniform(heading_low, heading_high)
    heading = np.deg2rad(heading_deg)
    # Primarily lateral across the table, with bounded fore/aft variation.
    direction = np.array([np.sin(heading), -np.cos(heading), 0.0], dtype=np.float64)
    latency_frames = 15 + (index % 4)  # 0.50--0.60 s observation/control latency.
    extension_frames = 22 + ((index * 5) % 8)  # 0.73--0.97 s arm motion.
    hold_frames = 18
    tuck = np.asarray(TUCK_TARGETS[(index // 2) % len(TUCK_TARGETS)], dtype=np.float64)
    return speed, heading_deg, direction, latency_frames, extension_frames, hold_frames, tuck


def npv(value):
    return value.detach().cpu().numpy()


def indices(names, requested):
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise RuntimeError(f"missing joints: {missing}")
    return [lookup[name] for name in requested]


def body_index(names, candidates):
    for name in candidates:
        if name in names:
            return names.index(name)
    raise RuntimeError(f"missing body: {candidates}")


def quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quat_rpy(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
        np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
    ], dtype=np.float32)


def matrix_rpy(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.array([
        np.arctan2(rotation[2, 1], rotation[2, 2]),
        np.arcsin(np.clip(-rotation[2, 0], -1, 1)),
        np.arctan2(rotation[1, 0], rotation[0, 0]),
    ], dtype=np.float32)


def quat_multiply(left, right):
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def rgb(scene, name):
    image = npv(scene[name].data.output["rgb"][0])[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Use the identical center-crop/resize path as the 960x540 real D435 data.
    if image.shape[:2] == (540, 960):
        image = np.asarray(
            Image.fromarray(image[:, 120:840]).resize((640, 480), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return image


def bind_brainco_dark_material():
    """Override embedded CAD materials so the sim matches the real dark hand."""
    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, "/World/Looks/BrainCoDark")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/BrainCoDark/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.018, 0.020, 0.024))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.02)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    hand_parts = ("base_link", "thumb", "index", "middle", "ring", "pinky")
    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        is_hand_link = (
            "/Robot/" in path
            and name.startswith(("left_", "right_"))
            and any(part in name for part in hand_parts)
            and (name.endswith("link") or name.endswith("tip"))
        )
        if not is_hand_link:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        bound += 1
    if bound == 0:
        raise RuntimeError("no BrainCo visual geometry accepted the dark material override")


def bind_bunny_low_friction_material():
    """Bind measured-table sliding material to every bunny collider."""
    material_path = "/World/Looks/BunnySlidingPhysics"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.06,
        dynamic_friction=0.04,
        restitution=0.02,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    # The nested helper returns the root prim result (the root is not itself a
    # collider) even when all child collision prims bind successfully.
    sim_utils.bind_physics_material(
        "/World/envs/env_0/Object", material_path, stronger_than_descendants=True
    )


def hide_left_manipulator_visuals():
    """Hide only left-arm rendering; retain its articulation and collisions."""
    stage = omni.usd.get_context().get_stage()
    prefixes = (
        "left_shoulder_", "left_elbow_", "left_wrist_", "left_base",
        "left_hand_", "left_thumb_", "left_index_", "left_middle_",
        "left_ring_", "left_pinky_",
    )
    hidden = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        if "/Robot/" not in path or not name.startswith(prefixes):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            hidden += 1
    if hidden == 0:
        raise RuntimeError("no left manipulator visuals were hidden")


def contract_joint19(pos, vel, command, left_idx, right_idx, waist_idx):
    """Keep BrainCo joints outside the frozen UniFoLM 19D embodiment contract."""
    zeros = np.zeros(2, dtype=np.float32)
    qpos = np.concatenate((pos[left_idx], pos[right_idx], zeros, pos[waist_idx])).astype(np.float32)
    qvel = np.concatenate((vel[left_idx], vel[right_idx], zeros, vel[waist_idx])).astype(np.float32)
    action = np.concatenate(
        (command[left_idx], command[right_idx], zeros, command[waist_idx])
    ).astype(np.float32)
    return qpos, qvel, action


def split_for_seed(seed):
    bucket = int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def solve_line(ik, q, start_xyz, end_xyz, steps):
    output = []
    rotation = ik.forward_kinematics(q)[:3, :3].copy()
    for xyz in np.linspace(start_xyz, end_xyz, steps + 1)[1:]:
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = xyz
        result = ik.solve(target, q)
        if not result.ok:
            raise RuntimeError(f"ik:{result.reason}")
        q = np.asarray(result.q_rad, dtype=np.float64)
        output.append(q.copy())
    return output, q


def trajectory(ik, start_q, object_base):
    start_xyz = ik.forward_kinematics(start_q)[:3, 3]
    raised = np.array([start_xyz[0], start_xyz[1], max(0.22, start_xyz[2] + 0.12)])
    # Proxy origin is at its feet. Approach its near torso face from robot-forward (-X object face).
    touch = np.array([object_base[0] - 0.052, object_base[1], object_base[2] + 0.020])
    above = np.array([touch[0] - 0.035, touch[1], max(0.22, touch[2] + 0.06)])
    pre = np.array([touch[0] - 0.025, touch[1], touch[2]])
    push = touch.copy()
    q = start_q.copy()
    path = []
    for destination, steps in ((raised, 24), (above, 24), (pre, 24), (push, 18)):
        source = ik.forward_kinematics(q)[:3, 3]
        segment, q = solve_line(ik, q, source, destination, steps)
        path.extend(segment)
    path.extend([q.copy()] * 12)
    source = ik.forward_kinematics(q)[:3, 3]
    retreat, q = solve_line(ik, q, source, raised, 24)
    path.extend(retreat)
    return path


def make_sample(env, left_idx, right_idx, waist_idx, full_command, ee_target, contact, safety):
    scene, robot = env.scene, env.scene["robot"]
    pos = npv(robot.data.joint_pos[0]).astype(np.float32)
    vel = npv(robot.data.joint_vel[0]).astype(np.float32)
    names = list(robot.body_names)
    lp = body_index(names, ("left_hand_palm_link",))
    rp = body_index(names, ("right_hand_palm_link",))
    bpos = npv(robot.data.body_pos_w[0]).astype(np.float32)
    bquat = npv(robot.data.body_quat_w[0]).astype(np.float32)
    root_pos = npv(robot.data.root_pos_w[0]).astype(np.float32)
    root_rot = quat_matrix(npv(robot.data.root_quat_w[0]))
    left = np.concatenate([
        root_rot.T @ (bpos[lp] - root_pos),
        matrix_rpy(root_rot.T @ quat_matrix(bquat[lp])),
    ])
    right = np.concatenate([
        root_rot.T @ (bpos[rp] - root_pos),
        matrix_rpy(root_rot.T @ quat_matrix(bquat[rp])),
    ])
    qpos, qvel, command_ordered = contract_joint19(
        pos, vel, full_command, left_idx, right_idx, waist_idx
    )
    ee = np.concatenate([left, right, qpos[[14, 15]], qpos[16:19]]).astype(np.float32)
    ee_target = np.asarray(ee_target, dtype=np.float32).copy()
    # The left arm is commanded to hold. Never label it with a zero/world-origin
    # EE target: that would teach unsafe coupled motion during right-arm touch.
    ee_target[:6] = left
    obj = scene["object"].data
    head = rgb(scene, "front_camera")
    images = {"cam_left_high": head, "cam_right_high": head.copy(),
              "cam_left_wrist": head.copy(), "cam_right_wrist": head.copy()}
    force = np.zeros(6, np.float32)
    force[:3] = root_rot.T @ contact
    object_position = root_rot.T @ (npv(obj.root_pos_w[0]).astype(np.float32) - root_pos)
    object_velocity = root_rot.T @ npv(obj.root_lin_vel_w[0]).astype(np.float32)
    return FrameSample(0.0, images, qpos, qvel, command_ordered,
                       ee, ee_target, object_position.astype(np.float32),
                       object_velocity.astype(np.float32), force,
                       tracking_valid=bool(np.max(np.abs(command_ordered[7:14]-qpos[7:14])) < 0.18),
                       grasped=False, safety_event=safety)


def main():
    for required in (args.bunny_usd, args.table_usd, args.robot_usd, args.robot_urdf):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rejected = args.output_dir / "rejected"
    rejected.mkdir(exist_ok=True)
    # Each worker owns its output directory, so leftovers here are necessarily
    # from an interrupted prior invocation rather than a live peer writer.
    for stale in args.output_dir.glob("*.partial.hdf5"):
        stale.unlink(missing_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.observations.policy.robot_joint_state.params = {"enable_dds": False}
    cfg.observations.policy.robot_gipper_state.params = {"enable_dds": False}
    cfg.observations.policy.camera_image = None
    # Dataset capture reads state directly. Stock rewards/terminations publish DDS
    # diagnostics and can auto-reset the scene, neither of which belongs here.
    cfg.rewards = None
    cfg.terminations = None
    # IK returns absolute joint angles. Do not add the stock task's default
    # joint pose a second time in the action manager.
    cfg.actions.joint_pos.use_default_offset = False
    cfg.scene.robot.spawn.usd_path = str(args.robot_usd.resolve())
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=0.002, rest_offset=0.0
    )
    cfg.scene.robot.init_state.joint_pos = {".*": 0.0}
    cfg.scene.robot.actuators["hands"].joint_names_expr = [
        ".*_(thumb_metacarpal|thumb_proximal|index_proximal|middle_proximal|"
        "ring_proximal|pinky_proximal)_joint"
    ]
    # The stock Dex1 task uses very soft arm gains and sags roughly 0.25 rad
    # under gravity. Match position-controlled G1 behavior closely enough that
    # camera state and action labels describe the same motion.
    cfg.scene.robot.actuators["arms"].stiffness = {
        ".*_shoulder_.*_joint": 300.0,
        ".*_elbow_joint": 300.0,
        ".*_wrist_.*_joint": 120.0,
    }
    cfg.scene.robot.actuators["arms"].damping = {
        ".*_shoulder_.*_joint": 12.0,
        ".*_elbow_joint": 12.0,
        ".*_wrist_.*_joint": 6.0,
    }
    # Wrist streams are absent from the real captures. Keep the frozen
    # four-camera storage contract by duplicating the head view in make_sample.
    cfg.scene.left_wrist_camera = None
    cfg.scene.right_wrist_camera = None
    cfg.scene.front_camera.width = 960
    cfg.scene.front_camera.height = 540
    # Match the D435 transport: the physical source updates at 60 Hz, while
    # policy samples below are timestamped/stored at 30 Hz. Isaac still runs
    # multiple physics ticks per policy sample, so no trajectory time is lost.
    cfg.scene.front_camera.update_period = 1.0 / 60.0
    if args.camera_variant != "head":
        direction = 1.0 if args.camera_variant == "angled-left" else -1.0
        base_position = np.asarray(cfg.scene.front_camera.offset.pos, dtype=np.float64)
        base_rotation = np.asarray(cfg.scene.front_camera.offset.rot, dtype=np.float64)
        yaw = np.deg2rad(8.0 * direction)
        local_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        cfg.scene.front_camera.offset.pos = tuple(base_position + np.array([0.0, 0.025 * direction, 0.0]))
        cfg.scene.front_camera.offset.rot = tuple(quat_multiply(base_rotation, local_yaw))
    # The stock warehouse has broken external MDL references and is unlike the
    # real lab. A simple bounded backdrop will be introduced after geometry
    # and contact pass the diagnostic gate.
    cfg.scene.room_walls = None
    cfg.scene.ground = AssetBaseCfg(
        prim_path="/World/LabFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.055, 0.058, 0.062), roughness=0.98, metallic=0.0
            ),
        ),
    )
    cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.84, 0.85, 0.86), intensity=600.0),
    )
    if args.diagnostic_design_index is not None:
        cfg.scene.dome_light.spawn.intensity = (500.0, 600.0, 720.0)[
            args.diagnostic_design_index % 3
        ]
    cfg.scene.packing_table.spawn.usd_path = str(args.table_usd.resolve())
    cfg.scene.object.spawn = UsdFileCfg(
        usd_path=str(args.bunny_usd),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.05,
            angular_damping=1.5,
            max_linear_velocity=0.5,
            max_angular_velocity=1.0,
            max_depenetration_velocity=0.2,
            sleep_threshold=0.005,
            stabilization_threshold=0.002,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0))
    cfg.scene.right_hand_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_BRAINCO_CONTACT_EXPR}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )
    # A second pair-filtered sensor makes "left arm is irrelevant" an
    # enforceable physical contract, not just a camera-framing assumption.
    cfg.scene.left_hand_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_BRAINCO_CONTACT_EXPR}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )
    cfg.scene.right_hand_table_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_BRAINCO_CONTACT_EXPR}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/PackingTable/CollisionTop"],
        update_period=0.0,
        history_length=3,
    )
    env = gym.make(args.task, cfg=cfg).unwrapped
    bind_brainco_dark_material()
    bind_bunny_low_friction_material()
    robot = env.scene["robot"]
    contact_body_names = list(env.scene["right_hand_contact"].body_names)
    left_contact_body_names = list(env.scene["left_hand_contact"].body_names)
    joint_names = list(robot.joint_names)
    left_idx, right_idx, waist_idx = (indices(joint_names, group) for group in (LEFT_ARM, RIGHT_ARM, WAIST))
    completed = (p for p in args.output_dir.glob("episode_*.hdf5") if ".partial." not in p.name)
    accepted = sum(1 for p in completed if not validate_episode(p))
    attempt = 0
    max_attempts = (
        args.episodes
        if args.rollout_output is not None
        else max(args.episodes, int(args.episodes * args.max_attempt_factor))
    )
    if args.rollout_output is not None:
        args.rollout_output.parent.mkdir(parents=True, exist_ok=True)
        args.rollout_output.unlink(missing_ok=True)
    t0 = time.monotonic()
    while (
        attempt < args.episodes
        if args.rollout_output is not None
        else accepted < args.episodes and attempt < max_attempts
    ):
        free_gb = shutil.disk_usage(args.output_dir).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise SystemExit(f"disk guard stopped generation: {free_gb:.1f} GiB free < {args.min_free_gb:.1f} GiB")
        seed = args.seed_start + attempt
        attempt += 1
        rng = random.Random(seed)
        final = args.output_dir / f"episode_{seed:06d}.hdf5"
        if final.exists() and not validate_episode(final):
            # Valid files were included in the initial accepted count above.
            # On resume, skip their seeds without counting them a second time.
            continue
        partial = final.with_suffix(".partial.hdf5")
        partial.unlink(missing_ok=True)
        writer = None
        reason = "unknown"
        rollout_metrics = None
        try:
            env.reset(seed=seed)
            # Spawn the unused left arm directly in its official side-rest
            # pose.  Interpolating there from the task's zero pose sweeps the
            # hand through the table edge, which physically traps the arm and
            # makes its controller appear not to track.
            initial_joint_pos = robot.data.default_joint_pos.clone()
            initial_joint_pos[0, left_idx] = torch.as_tensor(
                REAL_LEFT_HOLD, dtype=torch.float32, device=env.device
            )
            robot.write_joint_state_to_sim(
                initial_joint_pos,
                torch.zeros_like(robot.data.default_joint_vel),
            )
            full = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
            placement_root = npv(robot.data.root_pos_w[0])
            placement_rot = quat_matrix(npv(robot.data.root_quat_w[0]))
            speed, heading_deg, direction_base, latency_frames, extension_frames, hold_frames, tuck_base = design_parameters(seed, rng)
            root_pos = npv(robot.data.root_pos_w[0])
            root_rot = quat_matrix(npv(robot.data.root_quat_w[0]))
            intercept_base = np.array([
                rng.uniform(0.34, 0.37), rng.uniform(-0.14, -0.07),
                TABLE_Z - root_pos[2] + 0.095,
            ])
            reaction_time_s = (latency_frames + extension_frames) / 30.0
            # Center-to-center planning needs an extra radius allowance so the
            # plush does not collide with the extending hand halfway through.
            travel_m = speed * reaction_time_s + 0.071
            source_base = intercept_base.copy()
            source_base[:2] -= direction_base[:2] * travel_m
            # Preserve travel distance while translating long/high-speed
            # corridors away from the real table's raised perimeter rails.
            if source_base[1] > 0.11:
                shift = source_base[1] - 0.11
                source_base[1] -= shift
                intercept_base[1] -= shift
            elif source_base[1] < -0.22:
                shift = -0.22 - source_base[1]
                source_base[1] += shift
                intercept_base[1] += shift
            # Put the blocking palm into the oncoming corridor. Faster bins
            # receive a slightly larger lead because the hand and bunny move
            # concurrently; this changes contact timing, not the object path.
            blocking_contact_lead_m = 0.008 + max(0.0, speed - 0.10) * 0.50
            hand_intercept_base = (
                intercept_base - direction_base * blocking_contact_lead_m
            )
            if not (0.25 <= source_base[0] <= 0.41 and -0.28 <= source_base[1] <= 0.28):
                raise RuntimeError(f"launch_outside_table:{source_base.tolist()}")
            object_world = placement_root + placement_rot @ source_base
            object_world[2] = TABLE_Z + 0.002
            target_pose = torch.tensor(
                [[*object_world, 1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=env.device,
            )
            parking_pose = target_pose.clone()
            parking_pose[0, 2] = 2.0
            env.scene["object"].write_root_pose_to_sim(parking_pose)
            env.scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            for _ in range(20):
                env.scene["object"].write_root_pose_to_sim(parking_pose)
                env.scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
                env.step(full)
            safe_q = np.array(
                [-0.31766403, -0.07756469, -0.07918049, -0.30470770,
                 0.00263729, -0.11958560, -0.01605866],
                dtype=np.float32,
            )
            current_q = npv(robot.data.joint_pos[0, right_idx]).astype(np.float32)
            current_left = npv(robot.data.joint_pos[0, left_idx]).astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, 25)[1:]:
                q_clear = (1.0 - alpha) * current_q + alpha * safe_q
                q_left = (1.0 - alpha) * current_left + alpha * REAL_LEFT_HOLD
                full[0, right_idx] = torch.as_tensor(q_clear, dtype=torch.float32, device=env.device)
                full[0, left_idx] = torch.as_tensor(q_left, dtype=torch.float32, device=env.device)
                for _ in range(3):
                    env.scene["object"].write_root_pose_to_sim(parking_pose)
                    env.scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
                    env.step(full)
            # Geometry-sensitive cache key: changing the physical contact lead
            # must never silently reuse an IK plan from an older scene policy.
            lead_um = round(blocking_contact_lead_m * 1_000_000)
            plan_file = (
                args.output_dir / ".plans" / f"intercept_{seed:06d}_lead{lead_um:05d}um.npz"
            )
            if not plan_file.exists():
                command = [sys.executable, str(root / "scripts/plan_touch_ik.py"),
                           "--urdf", str(args.robot_urdf.resolve()),
                           "--ee-frame", "right_hand_palm_link",
                           "--start-q=" + ",".join(map(str, safe_q)),
                           "--tuck-base=" + ",".join(map(str, tuck_base)),
                           "--intercept-base=" + ",".join(map(str, hand_intercept_base)),
                           "--extension-frames", str(extension_frames),
                           "--hold-frames", str(hold_frames),
                           "--output", str(plan_file)]
                child_env = os.environ.copy()
                child_env["PYTHONPATH"] = str(root / "src")
                subprocess.run(command, cwd=root, env=child_env, check=True,
                               capture_output=True, text=True)
            planned = np.load(plan_file)
            path = planned["q"]
            ee_local = planned["ee_xyz"]
            tuck_q = planned["tuck_q"]
            tuck_xyz = planned["tuck_xyz"]
            extension_m = float(planned["extension_m"])
            # Pre-position beside the torso while the object is parked. This
            # motion is intentionally outside the demonstration.
            from_safe = npv(robot.data.joint_pos[0, right_idx]).astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, 30)[1:]:
                q_tuck = (1.0 - alpha) * from_safe + alpha * tuck_q
                full[0, right_idx] = torch.as_tensor(q_tuck, dtype=torch.float32, device=env.device)
                for _ in range(3):
                    env.scene["object"].write_root_pose_to_sim(parking_pose)
                    env.scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
                    env.step(full)
            pre_capture_error = float("inf")
            pre_capture_left_error = float("inf")
            for _ in range(180):
                env.step(full)
                actual_right = npv(robot.data.joint_pos[0, right_idx]).astype(np.float32)
                actual_left = npv(robot.data.joint_pos[0, left_idx]).astype(np.float32)
                pre_capture_error = float(np.max(np.abs(actual_right - tuck_q)))
                pre_capture_left_error = float(np.max(np.abs(actual_left - REAL_LEFT_HOLD)))
                if (
                    pre_capture_error <= args.pre_capture_tracking_tolerance
                    and pre_capture_left_error <= args.pre_capture_tracking_tolerance
                ):
                    break
            if (
                pre_capture_error > args.pre_capture_tracking_tolerance
                or pre_capture_left_error > args.pre_capture_tracking_tolerance
            ):
                raise RuntimeError(
                    f"pre_capture_tracking_error:right={pre_capture_error:.4f},"
                    f"left={pre_capture_left_error:.4f},"
                    f"left_actual={actual_left.tolist()},"
                    f"left_target={REAL_LEFT_HOLD.tolist()}"
                )
            env.scene["object"].write_root_pose_to_sim(target_pose)
            env.scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            for _ in range(90):
                env.step(full)
            object_actual = npv(env.scene["object"].data.root_pos_w[0]).astype(np.float32)
            settled_pose = env.scene["object"].data.root_pose_w[0].clone().unsqueeze(0)
            settle_delta = object_actual - np.asarray(object_world, dtype=np.float32)
            settle_motion = float(np.linalg.norm(settle_delta))
            launch_velocity_world = placement_rot @ (direction_base * speed)
            launch_velocity = torch.tensor(
                [[*launch_velocity_world, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device
            )
            drive_end_frame = latency_frames + extension_frames
            zero_wrench = torch.zeros((1, 1, 3), dtype=torch.float32, device=env.device)

            def set_motion_drive(active):
                if not active:
                    env.scene["object"].set_external_force_and_torque(
                        zero_wrench, zero_wrench, is_global=True
                    )
                    return 0.0
                velocity_base = root_rot.T @ npv(env.scene["object"].data.root_lin_vel_w[0])
                tangent_speed = float(np.dot(velocity_base, direction_base))
                # Feed-forward offsets real plush/table sliding resistance;
                # bounded feedback prevents drift without rewriting velocity.
                magnitude = float(np.clip(0.44 + 1.50 * (speed - tangent_speed), 0.0, 0.62))
                force_world = placement_rot @ (direction_base * magnitude)
                force = torch.tensor(force_world, dtype=torch.float32, device=env.device).view(1, 1, 3)
                angular_world = npv(env.scene["object"].data.root_ang_vel_w[0]).astype(np.float64)
                damping_torque = np.clip(-0.030 * angular_world, -0.035, 0.035)
                torque = torch.tensor(
                    damping_torque, dtype=torch.float32, device=env.device
                ).view(1, 1, 3)
                env.scene["object"].set_external_force_and_torque(
                    force, torque, is_global=True
                )
                return magnitude

            # Matched no-hand counterfactual: same settled pose, impulse, drive,
            # and horizon with the arm held tucked. It is run before capture.
            env.scene["object"].write_root_pose_to_sim(settled_pose)
            env.scene["object"].write_root_velocity_to_sim(launch_velocity)
            baseline_tangent_speeds = []
            baseline_frames = latency_frames + extension_frames + hold_frames + 20
            for baseline_frame in range(baseline_frames):
                set_motion_drive(True)
                for _ in range(3 + (1 if baseline_frame % 3 == 2 else 0)):
                    env.step(full)
                velocity_base = root_rot.T @ npv(env.scene["object"].data.root_lin_vel_w[0])
                baseline_tangent_speeds.append(float(np.dot(velocity_base, direction_base)))
            baseline_tangent_speeds = np.asarray(baseline_tangent_speeds, dtype=np.float32)
            baseline_reaction_speed = float(np.median(
                baseline_tangent_speeds[max(0, latency_frames-3):latency_frames+1]
            ))
            if baseline_reaction_speed < 0.60 * speed:
                raise RuntimeError(
                    f"friction_baseline_stalls_before_intercept:launch={speed:.4f},"
                    f"reaction={baseline_reaction_speed:.4f},final={baseline_tangent_speeds[-1]:.4f}"
                )
            set_motion_drive(False)
            # This reset is the final object state write. After launch begins,
            # the bunny is fully dynamic and never re-posed or velocity-driven.
            env.scene["object"].write_root_pose_to_sim(settled_pose)
            env.scene["object"].write_root_velocity_to_sim(launch_velocity)
            matrix = env.scene["right_hand_contact"].data.force_matrix_w
            initial_contact = (
                np.zeros(3, np.float32)
                if matrix is None
                else npv(matrix[0]).reshape(-1, 3).sum(axis=0).astype(np.float32)
            )
            rp = body_index(list(robot.body_names), ("right_hand_palm_link",))
            hand_position = npv(robot.data.body_pos_w[0, rp]).astype(np.float32)
            object_center = object_actual.copy()
            object_center[2] += 0.09
            initial_separation = float(np.linalg.norm(hand_position - object_center))
            initial_force = float(np.linalg.norm(initial_contact))
            if initial_separation < args.min_initial_separation or initial_force > 0.25:
                raise RuntimeError(
                    f"initial_scene:settle={settle_motion:.4f},delta={settle_delta.tolist()},"
                    f"z={object_actual[2]:.4f},separation={initial_separation:.4f},force={initial_force:.3f},"
                    f"hand={hand_position.tolist()},object_center={object_center.tolist()}"
                )
            command_path = [(tuck_q.copy(), tuck_xyz.copy())] * latency_frames
            command_path.extend((q.copy(), xyz.copy()) for q, xyz in zip(path, ee_local))
            writer = EpisodeWriter(partial, INSTRUCTION, metadata={
                "generator": "scripted_ik_moving_block_v9", "trajectory_seed": seed,
                "split": split_for_seed(seed), "smoke_test": False,
                "camera_variant": args.camera_variant,
                "source_task": args.task, "object_x": object_world[0], "object_y": object_world[1],
                "launch_speed_m_s": speed, "launch_heading_deg": heading_deg,
                "blocking_contact_lead_m": blocking_contact_lead_m,
                "latency_frames": latency_frames, "reaction_time_s": reaction_time_s,
                "source_camera_fps": 60.0, "policy_sample_fps": 30.0,
                "camera_resampling": "timestamped_even_frames_from_60hz_source",
                "extension_frames": extension_frames, "extension_m": extension_m,
                "fixed_base_balance_not_validated": True,
                "left_arm_visual_policy": "physical_fixed_hip_rest_out_of_head_view",
                "robot_urdf": str(args.robot_urdf.resolve()),
                "contact_semantics": "filtered_right_brainco_hand_to_object_two_physics_steps",
                "contact_body_names": json.dumps(contact_body_names),
                "left_contact_body_names": json.dumps(left_contact_body_names)})
            contact_frames = 0
            safety_frames = 0
            min_clearance = float("inf")
            min_enabled_clearance = float("inf")
            image_ok = True
            tracking_invalid_frames = 0
            max_left_target_error = 0.0
            max_eligible_contact_force = 0.0
            max_precontact_force = 0.0
            max_left_object_force = 0.0
            max_right_table_force = 0.0
            max_raise_displacement = 0.0
            early_interference = False
            max_action_step = 0.0
            previous_command = None
            initial_object = object_actual.copy()
            frame = 0
            contact_streak = 0
            raw_contact_seen = False
            contacting_links = []
            drive_forces_n = []
            palm_positions_base = []
            tangent_speeds = []
            object_positions_base = []
            contact_states = []
            lateral_speeds = []
            angular_speeds = []
            tilt_degrees = []
            first_contact_frame = None
            block_hold_scheduled = False
            table_exit = False
            early_contact = False
            while frame < len(command_path):
                q_target, target_xyz_local = command_path[frame]
                full[0, right_idx] = torch.as_tensor(q_target, dtype=torch.float32, device=env.device)
                drive_forces_n.append(set_motion_drive(not raw_contact_seen))
                pair_contact = np.zeros(3, np.float32)
                pair_forces = np.zeros((len(contact_body_names), 3), np.float32)
                for _ in range(3 + (1 if frame % 3 == 2 else 0)):
                    env.step(full)
                    matrix = env.scene["right_hand_contact"].data.force_matrix_w
                    if matrix is None:
                        pair_forces = np.zeros((len(contact_body_names), 3), np.float32)
                    else:
                        pair_forces = npv(matrix[0]).reshape(-1, 3).astype(np.float32)
                    pair_contact = pair_forces.sum(axis=0).astype(np.float32)
                    left_matrix = env.scene["left_hand_contact"].data.force_matrix_w
                    if left_matrix is not None:
                        left_forces = npv(left_matrix[0]).reshape(-1, 3).astype(np.float32)
                        if len(left_forces):
                            max_left_object_force = max(
                                max_left_object_force,
                                float(np.max(np.linalg.norm(left_forces, axis=1))),
                            )
                    table_matrix = env.scene["right_hand_table_contact"].data.force_matrix_w
                    if table_matrix is not None:
                        table_forces = npv(table_matrix[0]).reshape(-1, 3).astype(np.float32)
                        if len(table_forces):
                            max_right_table_force = max(
                                max_right_table_force,
                                float(np.max(np.linalg.norm(table_forces, axis=1))),
                            )
                    if float(np.linalg.norm(pair_contact)) > args.contact_force_threshold:
                        contact_streak += 1
                        raw_contact_seen = True
                    else:
                        contact_streak = 0
                eligible_contact = contact_streak >= 2
                contact = pair_contact if eligible_contact else np.zeros(3, np.float32)
                contacting_link = ""
                if eligible_contact and len(pair_forces):
                    contacting_link = contact_body_names[int(np.argmax(np.linalg.norm(pair_forces, axis=1)))]
                contacting_links.append(contacting_link)
                contact_states.append(eligible_contact)
                contact_excess = float(np.linalg.norm(contact))
                hand_position = npv(robot.data.body_pos_w[0, rp]).astype(np.float32)
                hand_z = float(hand_position[2])
                object_center = npv(env.scene["object"].data.root_pos_w[0]).astype(np.float32)
                object_center[2] += 0.09
                if eligible_contact:
                    contact_frames += 1
                    if first_contact_frame is None:
                        first_contact_frame = frame
                        early_contact = frame < latency_frames
                    max_eligible_contact_force = max(
                        max_eligible_contact_force, contact_excess
                    )
                if not raw_contact_seen:
                    max_precontact_force = max(max_precontact_force, contact_excess)
                    early_interference |= max_precontact_force > args.max_precontact_force
                clearance = hand_z - TABLE_Z
                min_clearance = min(min_clearance, clearance)
                if contact_frames == 0:
                    min_enabled_clearance = min(min_enabled_clearance, clearance)
                safety = contact_frames == 0 and clearance < 0.05
                safety_frames += int(safety)
                object_position_w = npv(env.scene["object"].data.root_pos_w[0]).astype(np.float32)
                object_position_base = root_rot.T @ (object_position_w - root_pos)
                object_positions_base.append(object_position_base.copy())
                object_velocity_base = root_rot.T @ npv(env.scene["object"].data.root_lin_vel_w[0])
                tangent = float(np.dot(object_velocity_base, direction_base))
                lateral = float(np.linalg.norm(object_velocity_base - tangent * direction_base))
                tangent_speeds.append(tangent)
                lateral_speeds.append(lateral)
                angular = float(np.linalg.norm(npv(env.scene["object"].data.root_ang_vel_w[0])))
                angular_speeds.append(angular)
                bunny_z = quat_matrix(npv(env.scene["object"].data.root_quat_w[0]))[:, 2]
                tilt_degrees.append(float(np.rad2deg(np.arccos(np.clip(bunny_z[2], -1.0, 1.0)))))
                table_exit |= not (
                    -0.46 <= object_position_base[0] <= 0.46
                    and -0.34 <= object_position_base[1] <= 0.34
                    and object_position_w[2] >= TABLE_Z - 0.015
                )
                full_command = npv(full[0])
                _, _, command = contract_joint19(
                    npv(robot.data.joint_pos[0]),
                    npv(robot.data.joint_vel[0]),
                    full_command,
                    left_idx,
                    right_idx,
                    waist_idx,
                )
                right_rotation_local = root_rot.T @ quat_matrix(npv(robot.data.body_quat_w[0, rp]))
                right_target = np.concatenate([target_xyz_local, matrix_rpy(right_rotation_local)])
                # Left target stays at its current measured pose; grippers and waist remain fixed.
                sample = make_sample(env, left_idx, right_idx, waist_idx, full_command, np.concatenate([
                    np.zeros(6, np.float32), right_target, command[[14, 15]], command[16:19]
                ]), contact, safety)
                palm_positions_base.append(sample.ee_qpos[6:9].copy())
                sample.timestamp = frame / 30.0
                tracking_invalid_frames += int(not sample.tracking_valid)
                max_left_target_error = max(
                    max_left_target_error,
                    float(np.linalg.norm(sample.ee_action[:3] - sample.ee_qpos[:3])),
                )
                if previous_command is not None:
                    max_action_step = max(
                        max_action_step,
                        float(np.max(np.abs(sample.action - previous_command))),
                    )
                previous_command = sample.action.copy()
                if frame in (0, len(path)//2, len(path)-1):
                    image_ok &= all(float(img.std()) > 5.0 for img in sample.images.values())
                phase = "observe moving plush" if frame < latency_frames else (
                    "extend right palm into path" if frame < latency_frames + extension_frames
                    else "hold blocking pose"
                )
                writer.append(sample, phase)
                if eligible_contact and not block_hold_scheduled:
                    block_hold_scheduled = True
                    hold = [(q_target.copy(), target_xyz_local.copy())] * 18
                    command_path = command_path[:frame + 1] + hold
                frame += 1
            displacement = float(np.linalg.norm(npv(env.scene["object"].data.root_pos_w[0]) - initial_object))
            tangent_speeds = np.asarray(tangent_speeds, dtype=np.float32)
            object_positions_base = np.asarray(object_positions_base, dtype=np.float32)
            lateral_speeds = np.asarray(lateral_speeds, dtype=np.float32)
            angular_speeds = np.asarray(angular_speeds, dtype=np.float32)
            tilt_degrees = np.asarray(tilt_degrees, dtype=np.float32)
            drive_forces_n = np.asarray(drive_forces_n, dtype=np.float32)
            palm_positions_base = np.asarray(palm_positions_base, dtype=np.float32)
            set_motion_drive(False)
            pre_contact_speed = post_contact_speed = rebound_speed = float("inf")
            baseline_post_speed = float("inf")
            sustained_stop_frames = 0
            stop_fraction = -float("inf")
            counterfactual_margin = -float("inf")
            actual_extension_m = 0.0
            post_contact_progress_m = float("inf")
            contact_dwell_s = 0.0
            if first_contact_frame is not None and first_contact_frame > 0:
                pre_start = max(0, first_contact_frame - 4)
                pre_contact_speed = float(np.median(tangent_speeds[pre_start:first_contact_frame]))
                actual_extension_m = float(np.linalg.norm(
                    palm_positions_base[first_contact_frame] - palm_positions_base[0]
                ))
                post_start = min(len(tangent_speeds), first_contact_frame + 6)
                post_stop = min(len(tangent_speeds), first_contact_frame + 16)
                if post_stop > post_start:
                    post_values = np.maximum(0.0, tangent_speeds[post_start:post_stop])
                    post_contact_speed = float(np.median(post_values))
                    rebound_speed = float(np.max(np.abs(tangent_speeds[post_start:post_stop])))
                    threshold = (1.0 - args.minimum_stop_fraction) * pre_contact_speed
                    sustained_stop_frames = int(np.sum(post_values <= threshold))
                    stop_fraction = 1.0 - post_contact_speed / max(pre_contact_speed, 1e-6)
                    if post_stop <= len(baseline_tangent_speeds):
                        baseline_post_speed = float(np.median(np.maximum(
                            0.0, baseline_tangent_speeds[post_start:post_stop]
                        )))
                        counterfactual_margin = baseline_post_speed - post_contact_speed
                progress_stop = min(
                    len(object_positions_base),
                    first_contact_frame + int(round(0.30 * 30.0)) + 1,
                )
                if progress_stop > first_contact_frame:
                    relative = (
                        object_positions_base[first_contact_frame:progress_stop]
                        - object_positions_base[first_contact_frame]
                    )
                    post_contact_progress_m = float(np.max(relative @ direction_base))
                longest_contact_run = 0
                current_contact_run = 0
                for active in contact_states[first_contact_frame:]:
                    current_contact_run = current_contact_run + 1 if active else 0
                    longest_contact_run = max(longest_contact_run, current_contact_run)
                contact_dwell_s = longest_contact_run / 30.0
            success = (
                contact_frames >= 1
                and safety_frames == 0
                and image_ok
                and tracking_invalid_frames == 0
                and args.minimum_reaction_s <= latency_frames / 30.0
                and max_left_target_error <= 0.03
                and max_action_step <= 0.08
                and max_eligible_contact_force <= 15.0
                and max_left_object_force <= 0.25
                and max_right_table_force <= args.maximum_right_table_force
                and not early_interference
                and not early_contact
                and not table_exit
                and pre_contact_speed >= 0.05
                and 0.15 <= actual_extension_m <= 0.30
                and stop_fraction >= args.minimum_stop_fraction
                and post_contact_speed <= args.maximum_postcontact_speed
                and sustained_stop_frames >= 6
                and rebound_speed <= 0.50 * pre_contact_speed
                and counterfactual_margin >= 0.30 * pre_contact_speed
                and float(np.max(angular_speeds[:max(1, first_contact_frame or 1)])) <= 0.12
                and float(np.max(tilt_degrees[:max(1, first_contact_frame or 1)])) <= 25.0
            )
            prohibited_contacts = int(
                safety_frames > 0
                or max_left_object_force > 0.25
                or max_right_table_force > args.maximum_right_table_force
                or early_interference
            )
            rollout_metrics = {
                "policy": "oracle_ik",
                "latency_ms": 0,
                "scenario_id": f"seed-{seed}",
                "seed": seed,
                "contact": contact_frames >= 1,
                "pre_contact_speed_m_s": pre_contact_speed,
                "post_contact_speed_m_s": post_contact_speed,
                "post_contact_progress_m": post_contact_progress_m,
                "contact_dwell_s": contact_dwell_s,
                "peak_impact_n": max_eligible_contact_force,
                "prohibited_contacts": prohibited_contacts,
                "failure_reason": "" if success else "physics_block_failed",
                "prediction_error_m": 0.0,
                "ik_rejected": False,
                "joint_limit_saturations": 0,
                "torque_saturations": 0,
                "safety_instrumentation_complete": False,
                "missing_safety_instrumentation": ["right_arm_self_contact"],
                "dataset_quality_gate": success,
                "launch_speed_m_s": speed,
                "launch_heading_deg": heading_deg,
            }
            writer.close(success=success)
            writer = None
            with h5py.File(partial, "r+") as h5:
                h5.attrs.update({"contact_frames": contact_frames, "min_palm_clearance_m": min_clearance,
                                 "min_enabled_palm_clearance_m": min_enabled_clearance,
                                 "object_displacement_m": displacement, "image_quality_ok": image_ok,
                                 "tracking_invalid_frames": tracking_invalid_frames,
                                 "max_left_target_error_m": max_left_target_error,
                                 "max_action_step_rad": max_action_step,
                                 "max_eligible_contact_force_n": max_eligible_contact_force,
                                 "initial_hand_object_separation_m": initial_separation,
                                 "initial_contact_force_n": initial_force,
                                 "settle_motion_m": settle_motion,
                                 "max_precontact_force_n": max_precontact_force,
                                 "maximum_left_hand_object_force_n": max_left_object_force,
                                 "maximum_right_hand_table_force_n": max_right_table_force,
                                 "maximum_allowed_postcontact_speed_m_s": args.maximum_postcontact_speed,
                                 "max_raise_displacement_m": max_raise_displacement,
                                 "early_interference": early_interference,
                                 "early_contact": early_contact,
                                 "first_contact_frame": -1 if first_contact_frame is None else first_contact_frame,
                                 "pre_contact_path_speed_m_s": pre_contact_speed,
                                 "post_contact_path_speed_m_s": post_contact_speed,
                                 "baseline_post_path_speed_m_s": baseline_post_speed,
                                 "stop_fraction": stop_fraction,
                                 "counterfactual_stop_margin_m_s": counterfactual_margin,
                                 "sustained_stop_frames": sustained_stop_frames,
                                 "maximum_rebound_path_speed_m_s": rebound_speed,
                                 "actual_palm_extension_at_contact_m": actual_extension_m,
                                 "maximum_angular_speed_rad_s": float(angular_speeds.max(initial=0.0)),
                                 "maximum_tilt_deg": float(tilt_degrees.max(initial=0.0)),
                                 "table_exit": table_exit,
                                 "launch_pose_rewritten_after_capture": False,
                                 "motion_drive_end_frame": drive_end_frame,
                                 "maximum_motion_drive_force_n": float(drive_forces_n.max(initial=0.0)),
                                 "object_anchored_until_contact": False,
                                 "pair_filtered_contact": True,
                                 "contact_force_threshold_n": args.contact_force_threshold,
                                 "pre_capture_tracking_error_rad": pre_capture_error,
                                 "pre_capture_left_tracking_error_rad": pre_capture_left_error})
                h5["sim_signals"].create_dataset(
                    "contacting_link",
                    data=np.asarray(contacting_links, dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
                h5["sim_signals"].create_dataset("path_tangent_speed", data=tangent_speeds)
                h5["sim_signals"].create_dataset("lateral_speed", data=lateral_speeds)
                h5["sim_signals"].create_dataset("angular_speed", data=angular_speeds)
                h5["sim_signals"].create_dataset("tilt_degrees", data=tilt_degrees)
                h5["sim_signals"].create_dataset("motion_drive_force_n", data=drive_forces_n)
                h5["sim_signals"].create_dataset(
                    "counterfactual_path_tangent_speed", data=baseline_tangent_speeds
                )
            errors = validate_episode(partial)
            if not success:
                errors.append(
                    "quality gate: "
                    f"contact={contact_frames} early={early_contact} safety={safety_frames} "
                    f"stop={stop_fraction:.3f} sustained={sustained_stop_frames} "
                    f"counterfactual_margin={counterfactual_margin:.4f} "
                    f"rebound={rebound_speed:.4f} extension={actual_extension_m:.4f} "
                    f"spin={angular_speeds.max(initial=0.0):.3f} table_exit={table_exit} image={image_ok}"
                )
            if errors:
                reason = "; ".join(errors)
                if len(list(rejected.glob("episode_*.hdf5"))) < args.keep_rejected:
                    partial.replace(rejected / partial.name.replace(".partial", ""))
                else:
                    partial.unlink(missing_ok=True)
            else:
                partial.replace(final)
                accepted += 1
                reason = "accepted"
        except Exception as exc:
            reason = f"exception:{type(exc).__name__}:{exc}"
            if rollout_metrics is None:
                rollout_metrics = {
                    "policy": "oracle_ik",
                    "latency_ms": 0,
                    "scenario_id": f"seed-{seed}",
                    "seed": seed,
                    "contact": False,
                    "pre_contact_speed_m_s": 0.0,
                    "post_contact_speed_m_s": float("inf"),
                    "post_contact_progress_m": float("inf"),
                    "contact_dwell_s": 0.0,
                    "peak_impact_n": float("inf"),
                    "prohibited_contacts": 0,
                    "failure_reason": reason,
                    "prediction_error_m": 0.0,
                    "ik_rejected": reason.startswith("exception:RuntimeError:ik:"),
                    "joint_limit_saturations": 0,
                    "torque_saturations": 0,
                    "safety_instrumentation_complete": False,
                    "missing_safety_instrumentation": ["right_arm_self_contact"],
                    "dataset_quality_gate": False,
                }
            if writer is not None:
                writer.close(success=False)
            if partial.exists():
                if len(list(rejected.glob("episode_*.hdf5"))) < args.keep_rejected:
                    partial.replace(rejected / partial.name.replace(".partial", ""))
                else:
                    partial.unlink(missing_ok=True)
        record = {"time": time.time(), "seed": seed, "accepted": accepted, "attempt": attempt, "result": reason}
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if args.rollout_output is not None:
            assert rollout_metrics is not None
            rollout_metrics["result"] = reason
            with args.rollout_output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rollout_metrics, sort_keys=True) + "\n")
        elapsed = time.monotonic() - t0
        rate = accepted / elapsed if elapsed else 0.0
        eta = (args.episodes - accepted) / rate if rate else None
        print(json.dumps({**record, "elapsed_s": round(elapsed, 1), "eta_s": None if eta is None else round(eta)}), flush=True)
    env.close()
    failure = None
    if args.rollout_output is None and accepted < args.episodes:
        failure = f"stopped after {attempt} attempts with only {accepted}/{args.episodes} accepted"
    simulation_app.close()
    if failure is not None:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
