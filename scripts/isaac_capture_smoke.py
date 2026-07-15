#!/usr/bin/env python3
"""Capture a short schema smoke episode from Unitree's stock Isaac Lab task.

Run from the root of a `unitree_sim_isaaclab` checkout so its `tasks` package is
importable. The output is deliberately tagged `smoke_test=true` and must not be
used for training. A single stock head view is duplicated into the two reference
dataset head-camera slots; the production bunny task must provide distinct views.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Stack-RgyBlock-G129-Dex1-Joint")
parser.add_argument("--output", type=Path)
parser.add_argument("--frames", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output is None:
    parser.error("--output is required")
args.enable_cameras = True

project_root = os.environ.get("G1_BUNNY_PROJECT_ROOT")
if not project_root:
    raise SystemExit("Set G1_BUNNY_PROJECT_ROOT to the humanoid-robot-grasping checkout")
unitree_sim_root = Path(os.environ.get("UNITREE_SIM_ROOT", Path.cwd()))
if not (unitree_sim_root / "tasks").is_dir():
    raise SystemExit(f"UNITREE_SIM_ROOT does not contain a tasks package: {unitree_sim_root}")
sys.path.insert(0, str(Path(project_root) / "src"))
sys.path.insert(0, str(unitree_sim_root))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tasks  # noqa: E402,F401  pylint: disable=unused-import
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from g1_bunny_vla.contract import CAMERA_NAMES, FrameSample  # noqa: E402
from g1_bunny_vla.episode_writer import EpisodeWriter  # noqa: E402


LEFT_ARM = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM = tuple(name.replace("left_", "right_") for name in LEFT_ARM)
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def tensor_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def find_indices(names: list[str], requested: tuple[str, ...]) -> list[int]:
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise RuntimeError(f"articulation is missing joints: {missing}")
    return [lookup[name] for name in requested]


def find_body_index(names: list[str], candidates: tuple[str, ...]) -> int:
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    for index, name in enumerate(names):
        if any(candidate in name for candidate in candidates):
            return index
    raise RuntimeError(f"none of the body candidates exist: {candidates}")


def rgb(scene, camera_name: str) -> np.ndarray:
    value = tensor_numpy(scene[camera_name].data.output["rgb"][0])
    value = value[..., :3]
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and value.max(initial=0) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def capture(scene, timestamp: float) -> FrameSample:
    robot = scene["robot"]
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    left_indices = find_indices(joint_names, LEFT_ARM)
    right_indices = find_indices(joint_names, RIGHT_ARM)
    waist_indices = find_indices(joint_names, WAIST)
    right_gripper_index = find_indices(joint_names, ("right_hand_Joint1_1",))[0]
    left_gripper_index = find_indices(joint_names, ("left_hand_Joint1_1",))[0]
    ordered = left_indices + right_indices + [right_gripper_index, left_gripper_index] + waist_indices

    joint_position = tensor_numpy(robot.data.joint_pos[0]).astype(np.float32)
    joint_velocity = tensor_numpy(robot.data.joint_vel[0]).astype(np.float32)
    qpos = joint_position[ordered]
    qvel = joint_velocity[ordered]

    left_body = find_body_index(body_names, ("left_hand_base_link", "left_wrist_yaw_link"))
    right_body = find_body_index(body_names, ("right_hand_base_link", "right_wrist_yaw_link"))
    body_position = tensor_numpy(robot.data.body_pos_w[0]).astype(np.float32)
    body_quaternion = tensor_numpy(robot.data.body_quat_w[0]).astype(np.float32)
    left_pose = np.concatenate(
        (body_position[left_body], quaternion_wxyz_to_rpy(body_quaternion[left_body]))
    )
    right_pose = np.concatenate(
        (body_position[right_body], quaternion_wxyz_to_rpy(body_quaternion[right_body]))
    )
    ee = np.concatenate(
        (
            left_pose,
            right_pose,
            qpos[[14, 15]],
            qpos[16:19],
        )
    ).astype(np.float32)

    object_key = "red_block" if "red_block" in scene.keys() else "object"
    object_data = scene[object_key].data
    plush_position = tensor_numpy(object_data.root_pos_w[0]).astype(np.float32)
    plush_velocity = tensor_numpy(object_data.root_lin_vel_w[0]).astype(np.float32)
    head = rgb(scene, "front_camera")
    images = {
        "cam_left_high": head,
        "cam_right_high": head.copy(),
        "cam_left_wrist": rgb(scene, "left_wrist_camera"),
        "cam_right_wrist": rgb(scene, "right_wrist_camera"),
    }
    assert set(images) == set(CAMERA_NAMES)
    return FrameSample(
        timestamp=timestamp,
        images=images,
        qpos=qpos,
        qvel=qvel,
        action=qpos.copy(),
        ee_qpos=ee,
        ee_action=ee.copy(),
        plush_position=plush_position,
        plush_velocity=plush_velocity,
    )


def main() -> int:
    env = None
    writer = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        writer = EpisodeWriter(
            args.output,
            "Hold position while observing the object.",
            metadata={
                "smoke_test": True,
                "source_task": args.task,
                "right_head_source": "duplicated_front_camera",
            },
        )
        action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
        sim_time = 0.0
        next_capture = 0.0
        capture_period = 1.0 / 30.0
        frame = 0
        step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        while frame < args.frames:
            env.step(action)
            sim_time += step_dt
            if sim_time + 1e-9 < next_capture:
                continue
            writer.append(capture(env.scene, frame * capture_period), "schema smoke capture")
            frame += 1
            next_capture = frame * capture_period
        writer.close(success=True)
        writer = None
        print(f"Wrote {frame} frames to {args.output}")
        return 0
    finally:
        if writer is not None:
            writer.close(success=False)
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
