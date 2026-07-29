#!/usr/bin/env python3
"""CPU rigid-body preflight for the G1 bridge's lift and gravity hold.

Run with ``uv run --with mujoco python scripts/sim/mujoco-g1-gravity-preflight.py``.
This is deliberately transport-free: it cannot command a physical robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmCommand,
    ArmState,
    RobotState,
)
from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.gravity import UrdfGravityCompensator
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path
from object_tracking.arm_tracking.joints import (
    BODY_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_LIMITS,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_GRAVITY_FF_LIMITS_NM,
    RIGHT_ARM_JOINT_NAMES,
)


START_RIGHT_Q = (
    0.2891673744,
    -0.1298251152,
    0.0039188415,
    0.9780925512,
    -0.1113813892,
    -0.0022170816,
    -0.0082091941,
)
SUPPORT_ORIGIN = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
SUPPORT_U = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
SUPPORT_V = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))


def body_state() -> tuple[float, ...]:
    return (
        -0.05,
        0.0,
        0.0,
        0.2,
        -0.15,
        0.0,
        -0.05,
        0.0,
        0.0,
        0.2,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *START_RIGHT_Q,
    )


def captured_support() -> SupportRegion:
    normal = -np.cross(SUPPORT_U, SUPPORT_V)
    normal /= np.linalg.norm(normal)
    return SupportRegion(
        plane=Plane(normal, -float(normal @ SUPPORT_ORIGIN)),
        origin=SUPPORT_ORIGIN,
        axis_u=SUPPORT_U,
        axis_v=SUPPORT_V,
        minimum_uv=(0.0, -0.3545687169),
        maximum_uv=(0.34, 0.3254312831),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated_pixel_near_edge"),),
        lateral_margin_m=0.07,
        source="captured_lab",
    )


class FakeClock:
    def __init__(self) -> None:
        self.monotonic = 10.0
        self.wall = 1_800_000_000.0

    def advance(self, dt: float) -> None:
        self.monotonic += dt
        self.wall += dt


class MujocoArmHardware:
    def __init__(
        self,
        mujoco: Any,
        model: Any,
        data: Any,
        clock: FakeClock,
        initial_body_q: tuple[float, ...],
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.clock = clock
        self.initial_body_q = initial_body_q
        self.joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in RIGHT_ARM_JOINT_NAMES
        }
        if any(value < 0 for value in self.joint_ids.values()):
            raise RuntimeError("MuJoCo model is missing a canonical G1 body joint")
        self.command: ArmCommand | None = None
        self.published = 0

    def start(self) -> None:
        pass

    def latest_state(self) -> RobotState:
        q_values = list(self.initial_body_q)
        dq_values = [0.0] * 29
        for offset, name in enumerate(RIGHT_ARM_JOINT_NAMES):
            joint = self.joint_ids[name]
            q_values[22 + offset] = float(
                self.data.qpos[self.model.jnt_qposadr[joint]]
            )
            dq_values[22 + offset] = float(
                self.data.qvel[self.model.jnt_dofadr[joint]]
            )
        q = tuple(q_values)
        dq = tuple(dq_values)
        arm_indices = tuple(range(15, 29))
        return RobotState(
            arm_q=tuple(q[index] for index in arm_indices),
            arm_dq=tuple(dq[index] for index in arm_indices),
            body_q=q,
            body_dq=dq,
            waist_q=tuple(q[12:15]),
            received_at=self.clock.monotonic,
            standing=True,
            standing_since=0.0,
            compatible_motion_mode=True,
            controller_available=True,
            mode_machine=5,
        )

    def publish(self, command: ArmCommand) -> None:
        self.command = command
        self.published += 1

    def apply_torques(self) -> None:
        if self.command is None:
            return
        arm_command = dict(
            zip((*LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES), range(14))
        )
        for arm_index, name in enumerate(RIGHT_ARM_JOINT_NAMES, start=7):
            joint_id = self.joint_ids[name]
            q_address = self.model.jnt_qposadr[joint_id]
            v_address = self.model.jnt_dofadr[joint_id]
            q = float(self.data.qpos[q_address])
            index = arm_command[name]
            assert index == arm_index
            sdk_torque = (
                self.command.kp[index] * (self.command.q[index] - q)
                + self.command.kd[index] * self.command.dq[index]
                + self.command.tau[index]
            )
            baseline_torque = (
                80.0 * (self.initial_body_q[22 + arm_index - 7] - q)
                + float(self.data.qfrc_bias[v_address])
            )
            torque = (
                self.command.weight * sdk_torque
                + (1.0 - self.command.weight) * baseline_torque
            )
            self.data.qfrc_applied[v_address] = torque

    def close(self) -> None:
        pass


def load_model(
    mujoco: Any,
    urdf: Path,
    timestep_s: float,
    *,
    belt_bunny_xyz_m: tuple[float, float, float] | None = None,
    bunny_radius_m: float = 0.055,
    bunny_mass_kg: float = 0.15,
) -> tuple[Any, Any]:
    xml = urdf.read_text(encoding="utf-8").replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        '<compiler discardvisual="false"/>',
    )
    assets = {
        f"meshes/{path.name}": path.read_bytes()
        for path in (urdf.parent / "meshes").iterdir()
        if path.is_file()
    }
    spec = mujoco.MjSpec.from_string(xml, assets)
    spec.assets = assets
    # The physical G1's low-level controller already stabilizes the base,
    # legs, waist, and opposite arm. Lock those joints and retain the complete
    # right-arm/hand inertial tree so this test isolates arm_sdk dynamics.
    for joint in tuple(spec.joints):
        if joint.name and joint.name not in RIGHT_ARM_JOINT_NAMES:
            spec.delete(joint)
    if belt_bunny_xyz_m is not None:
        bunny = spec.worldbody.add_body()
        bunny.name = "belt_bunny"
        bunny.pos = np.asarray(belt_bunny_xyz_m, dtype=float)
        lane = bunny.add_joint()
        lane.name = "belt_bunny_lane"
        lane.type = mujoco.mjtJoint.mjJNT_SLIDE
        lane.axis = np.asarray((0.0, 1.0, 0.0))
        lane.damping[0] = 0.001
        bunny_geom = bunny.add_geom()
        bunny_geom.name = "belt_bunny_geom"
        bunny_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        bunny_geom.size[0] = bunny_radius_m
        bunny_geom.mass = bunny_mass_kg

        table = spec.worldbody.add_body()
        table.name = "tabletop"
        table.pos = np.asarray((0.43, 0.06, 0.0025))
        table_geom = table.add_geom()
        table_geom.name = "tabletop_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = np.asarray((0.17, 0.34, 0.01))
    spec.option.timestep = timestep_s
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = spec.compile()
    # This first preflight isolates torque tracking and gravity hold. URDF
    # collision meshes include adjacent-link contacts that require the
    # upstream Unitree exclusion table, so contact is disabled here and tested
    # separately by the Isaac scene.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    if belt_bunny_xyz_m is not None:
        # Enable only the wrist-yaw/Dex hand subtree against the lane bunny
        # and table. This avoids the adjacent-link URDF contacts that Unitree's
        # upstream exclusion table normally filters while retaining real
        # contact impulses at the demo interaction surfaces.
        hand_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_wrist_yaw_link",
        )
        for geometry_index in range(model.ngeom):
            if model.geom_bodyid[geometry_index] == hand_body:
                model.geom_contype[geometry_index] = 1
                model.geom_conaffinity[geometry_index] = 2 | 4
        bunny_geometry = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "belt_bunny_geom",
        )
        table_geometry = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "tabletop_geom",
        )
        model.geom_contype[bunny_geometry] = 2
        model.geom_conaffinity[bunny_geometry] = 1
        model.geom_contype[table_geometry] = 4
        model.geom_conaffinity[table_geometry] = 1
    # The arm_sdk derivative term is joint-local. Encoding its -kd*dq half as
    # passive damping lets MuJoCo integrate that stiff term implicitly; the
    # desired-velocity half remains in the applied command above.
    model.dof_damping[:] = 3.0
    if belt_bunny_xyz_m is not None:
        bunny_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "belt_bunny_lane",
        )
        model.dof_damping[model.jnt_dofadr[bunny_joint]] = 0.001
    data = mujoco.MjData(model)
    return model, data


def set_joint_positions(
    mujoco: Any,
    model: Any,
    data: Any,
    values: tuple[float, ...],
) -> None:
    desired = dict(zip(BODY_JOINT_NAMES, values))
    for name in RIGHT_ARM_JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = desired[name]
    mujoco.mj_forward(model, data)


def gravity_crosscheck(
    mujoco: Any,
    model: Any,
    data: Any,
    compensator: UrdfGravityCompensator,
    initial_q: tuple[float, ...],
    *,
    samples: int = 250,
) -> dict[str, float | int]:
    """Compare the bounded robot-local evaluator with independent MuJoCo bias."""

    rng = np.random.default_rng(20260728)
    errors: list[float] = []
    limits = np.asarray(RIGHT_ARM_GRAVITY_FF_LIMITS_NM)
    for _ in range(samples):
        right_q = tuple(
            rng.uniform(lower + 0.08, upper - 0.08)
            for lower, upper in DEFAULT_RIGHT_JOINT_LIMITS
        )
        body_q = list(initial_q)
        body_q[22:29] = right_q
        set_joint_positions(mujoco, model, data, tuple(body_q))
        expected = np.asarray(
            [
                data.qfrc_bias[
                    model.jnt_dofadr[
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                ]
                for name in RIGHT_ARM_JOINT_NAMES
            ]
        )
        expected = np.clip(expected, -limits, limits)
        actual = np.asarray(compensator.torque(right_q))
        errors.extend(np.abs(actual - expected).tolist())
    set_joint_positions(mujoco, model, data, initial_q)
    return {
        "poses": samples,
        "joint_samples": len(errors),
        "maximum_absolute_error_nm": max(errors),
        "p95_absolute_error_nm": float(np.percentile(errors, 95)),
        "mean_absolute_error_nm": float(np.mean(errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--max-acceleration", type=float, default=4.0)
    parser.add_argument("--max-jerk", type=float, default=30.0)
    parser.add_argument("--without-gravity-ff", action="store_true")
    args = parser.parse_args()
    if (
        args.duration <= args.hold
        or args.physics_hz < 250.0
        or args.physics_hz % 250.0 != 0.0
    ):
        parser.error("duration must exceed hold and physics-hz must be a multiple of 250")

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is optional; run with: uv run --with mujoco python "
            "scripts/sim/mujoco-g1-gravity-preflight.py"
        ) from exc

    root = Path(__file__).resolve().parents[2]
    urdf = default_urdf_path(root)
    dt = 1.0 / args.physics_hz
    model, data = load_model(mujoco, urdf, dt)
    initial_q = body_state()
    set_joint_positions(mujoco, model, data, initial_q)

    support = captured_support()
    solver = G1RightArmIK(urdf)
    approach = solver.plan_guided_clearance(
        START_RIGHT_Q,
        support_plane=support,
        lift_m=0.12,
        forward_m=0.04,
    )
    if not approach.ok or not approach.q_path:
        raise RuntimeError(f"guided lift planning failed: {approach.reason}")
    target_q = tuple(float(value) for value in approach.q_path[-1])

    clock = FakeClock()
    hardware = MujocoArmHardware(mujoco, model, data, clock, initial_q)
    gravity_model = UrdfGravityCompensator(urdf)
    gravity_validation = gravity_crosscheck(
        mujoco,
        model,
        data,
        gravity_model,
        initial_q,
    )
    gravity = None if args.without_gravity_ff else gravity_model
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            allow_movement=True,
            calibration_id="lab-sim",
            control_hz=250.0,
            target_ttl_s=args.duration + 2.0,
            deadman_s=args.duration + 2.0,
            stable_standing_s=0.01,
            startup_settle_s=0.0,
            startup_settle_timeout_s=1.0,
            weight_ramp_s=0.10,
            max_velocity_rad_s=args.max_velocity,
            max_acceleration_rad_s2=args.max_acceleration,
            max_jerk_rad_s3=args.max_jerk,
            max_following_error_rad=1.0,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
        gravity_compensator=gravity,
    )
    controller.enable(session_id="mujoco-preflight", calibration_id="lab-sim")
    sequence = 0
    tracking_errors: list[float] = []
    elbow_errors: list[float] = []
    max_speed = 0.0
    target_sent = False
    started = time.perf_counter()
    steps = int(round(args.duration / dt))
    control_stride = int(round(args.physics_hz / 250.0))
    for step in range(steps):
        if step % control_stride == 0:
            controller.tick(clock.monotonic)
            if controller.state is ArmState.ARMED and not target_sent:
                sequence += 1
                controller.set_target(
                    session_id="mujoco-preflight",
                    sequence=sequence,
                    calibration_id="lab-sim",
                    right_arm_q=target_q,
                    pipeline_age_ms=0,
                )
                target_sent = True
        hardware.apply_torques()
        mujoco.mj_step(model, data)
        clock.advance(dt)
        state = hardware.latest_state()
        max_speed = max(max_speed, max(abs(value) for value in state.arm_dq[7:]))
        if target_sent and step * dt >= args.duration - args.hold:
            errors = [
                abs(actual - expected)
                for actual, expected in zip(state.arm_q[7:], target_q)
            ]
            tracking_errors.append(max(errors))
            elbow_errors.append(errors[3])
        if controller.state is ArmState.FAULT:
            break

    final = hardware.latest_state()
    final_errors = [
        abs(actual - expected) for actual, expected in zip(final.arm_q[7:], target_q)
    ]
    report = {
        "schema_version": 1,
        "simulator": f"mujoco-{mujoco.__version__}",
        "model": "g1_body29_hand14.urdf",
        "transport_free": True,
        "physics_hz": args.physics_hz,
        "gravity_feedforward": gravity is not None,
        "gravity_model_crosscheck": gravity_validation,
        "bridge_state": controller.state.value,
        "fault_reason": controller.fault_reason,
        "target_sent": target_sent,
        "published_commands": hardware.published,
        "maximum_measured_speed_rad_s": max_speed,
        "final_max_tracking_error_rad": max(final_errors),
        "final_elbow_error_rad": final_errors[3],
        "hold_max_tracking_error_rad": (
            None if not tracking_errors else max(tracking_errors)
        ),
        "hold_max_elbow_error_rad": None if not elbow_errors else max(elbow_errors),
        "bridge_metrics": controller.metrics.as_dict(),
        "wall_runtime_s": time.perf_counter() - started,
    }
    report["passed"] = bool(
        controller.state is ArmState.ARMED
        and target_sent
        and tracking_errors
        and gravity_validation["maximum_absolute_error_nm"] <= 1e-6
        and max(tracking_errors) <= 0.12
        and max(elbow_errors) <= 0.10
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
