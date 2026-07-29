from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmCommand,
    ArmState,
    RobotState,
)
from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path
from object_tracking.arm_tracking.joints import joint_contract_id
from object_tracking.arm_tracking.sim_closed_loop import (
    ObjectObservation,
    SimCommand,
    SimState,
)
from object_tracking.arm_tracking.sim_ipc import LatestPlannerProcess


START_Q = (
    0.2891673744,
    -0.1298251152,
    0.0039188415,
    0.9780925512,
    -0.1113813892,
    -0.0022170816,
    -0.0082091941,
)

ISOLATED_WORKER_WRAPPER = """
import importlib.abc, runpy, sys
blocked = (
    "unitree_sdk2py", "cyclonedds", "rclpy",
    "object_tracking.ros2_tracking", "object_tracking.ros2_transport",
    "object_tracking.arm_tracking.arm_unitree",
    "object_tracking.arm_tracking.depth_tcp",
)
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("forbidden simulator import: " + fullname)
        return None
def audit(event, args):
    if event in ("socket.connect", "socket.bind"):
        raise RuntimeError("simulator worker attempted network access: " + event)
sys.meta_path.insert(0, Blocker())
sys.addaudithook(audit)
script, project, config = sys.argv[1:]
sys.argv = [
    script, "--project-root", project, "--intercept-config", config,
]
runpy.run_path(script, run_name="__main__")
"""


class FakeClock:
    def __init__(self) -> None:
        self.monotonic = 10.0
        self.wall = 1_800_000_000.0

    def advance(self, dt: float) -> None:
        self.monotonic += dt
        self.wall += dt


class IdealPlant:
    """Apply every bridge command as the next 250 Hz measured state."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.state = self._state((0.0,) * 7 + START_Q, (0.0,) * 14)
        self.command: ArmCommand | None = None
        self.command_count = 0

    def _state(
        self,
        q: tuple[float, ...],
        dq: tuple[float, ...],
    ) -> RobotState:
        return RobotState(
            arm_q=q,
            arm_dq=dq,
            received_at=self.clock.monotonic,
            standing=True,
            standing_since=0.0,
            compatible_motion_mode=True,
            controller_available=True,
            mode_machine=5,
        )

    def start(self) -> None:
        pass

    def latest_state(self) -> RobotState:
        return self.state

    def publish(self, command: ArmCommand) -> None:
        self.command = command
        self.command_count += 1

    def apply(self) -> None:
        if self.command is None:
            q, dq = self.state.arm_q, self.state.arm_dq
        else:
            q, dq = self.command.q, self.command.dq
        self.state = self._state(q, dq)

    def close(self) -> None:
        pass


def captured_support() -> SupportRegion:
    origin = np.asarray((0.2622941631, 0.0478690107, 0.0129425348))
    axis_u = np.asarray((0.9993987830, -0.0065124773, 0.0340537822))
    axis_v = np.asarray((-0.0064480827, -0.9999772100, -0.0020004504))
    normal = -np.cross(axis_u, axis_v)
    return SupportRegion(
        plane=Plane(normal, -float(normal @ origin)),
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        minimum_uv=(0.0, -0.3545687169),
        maximum_uv=(0.34, 0.3254312831),
        certified_edges=("u_min",),
        edge_sources=(("u_min", "calibrated_pixel_near_edge"),),
        lateral_margin_m=0.07,
        source="captured_lab",
    )


def wait_for_command(
    client: LatestPlannerProcess,
    sequence: int,
    *,
    timeout_s: float = 90.0,
) -> SimCommand:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        command = client.latest_command()
        if command is not None and command.state_sequence == sequence:
            return command
        assert client.metrics().last_error is None
        time.sleep(0.005)
    raise AssertionError(f"planner did not answer state {sequence}")


def make_state(
    *,
    plant: IdealPlant,
    support: SupportRegion,
    sequence: int,
    simulation_time_s: float,
    with_observation: bool = True,
) -> SimState:
    return SimState(
        episode_id="captured-bridge-loop",
        sequence=sequence,
        simulation_time_s=simulation_time_s,
        calibration_id="lab-sim",
        joint_contract_id=joint_contract_id(),
        body_q_rad=plant.state.body_q,
        body_dq_rad_s=plant.state.body_dq,
        support_region=support,
        object_observation=(
            ObjectObservation(
                track_id=1,
                class_name="bunny",
                confidence=0.9,
                position_m=(0.36, 0.30 - 0.05 * simulation_time_s, 0.15),
                velocity_m_s=(0.0, -0.05, 0.0),
                observation_time_s=simulation_time_s,
                consecutive_observations=sequence + 1,
                residual_m=0.0,
            )
            if with_observation
            else None
        ),
    )


def test_actual_worker_drives_bridge_at_250_hz_without_path_faults() -> None:
    root = Path(__file__).resolve().parents[1]
    support = captured_support()
    solver = G1RightArmIK(default_urdf_path(root))
    clock = FakeClock()
    plant = IdealPlant(clock)
    controller = ArmBridgeController(
        plant,
        ArmBridgeConfig(
            allow_movement=True,
            calibration_id="lab-sim",
            control_hz=250.0,
            target_ttl_s=0.5,
            deadman_s=0.75,
            stable_standing_s=0.01,
            startup_settle_s=0.0,
            startup_settle_timeout_s=1.0,
            weight_ramp_s=0.01,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=4.0,
            max_jerk_rad_s3=30.0,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
    )
    controller.enable(session_id="sim", calibration_id="lab-sim")
    dt = 1.0 / 250.0
    for _ in range(8):
        controller.tick()
        plant.apply()
        clock.advance(dt)
    assert controller.state is ArmState.ARMED

    client = LatestPlannerProcess(
        (
            sys.executable,
            "-u",
            "-c",
            ISOLATED_WORKER_WRAPPER,
            str(root / "scripts/sim/g1-closed-loop-planner.py"),
            str(root),
            str(root / "tests/fixtures/lab-intercept-sim.yaml"),
        ),
        cwd=root,
    )
    client.start()
    sequence = 1
    target_sequence = 0
    submitted_at = 0.0
    applied_planner_sequence = -1
    reasons: list[str] = []
    maximum_speed = 0.0
    previous_acceleration = (0.0,) * 7
    approach_completed = False
    dropped_observations = 0
    try:
        client.submit(
            make_state(
                plant=plant,
                support=support,
                sequence=sequence,
                simulation_time_s=0.0,
            )
        )
        first = wait_for_command(client, sequence)
        assert first.status == "target"
        assert first.right_arm_q_rad is not None
        assert first.reason.startswith("preview_stage:adaptive_table_approach")

        simulation_time_s = 0.0
        started_wall = time.monotonic()
        while simulation_time_s < 8.0:
            if simulation_time_s - submitted_at >= 1.0 / 30.0:
                sequence += 1
                in_dropout = 1.0 <= simulation_time_s < 1.12
                dropped_observations += int(in_dropout)
                client.submit(
                    make_state(
                        plant=plant,
                        support=support,
                        sequence=sequence,
                        simulation_time_s=simulation_time_s,
                        with_observation=not in_dropout,
                    )
                )
                submitted_at = simulation_time_s

            latest = client.latest_command()
            if latest is not None and latest.state_sequence > applied_planner_sequence:
                applied_planner_sequence = latest.state_sequence
                reasons.append(latest.reason)
                if latest.status == "target":
                    assert latest.right_arm_q_rad is not None
                    target_sequence += 1
                    source_age_ms = int(
                        round(
                            1000.0
                            * max(
                                0.0,
                                simulation_time_s
                                - (
                                    simulation_time_s
                                    if latest.source_observation_time_s is None
                                    else latest.source_observation_time_s
                                ),
                            )
                        )
                    )
                    controller.set_target(
                        session_id="sim",
                        sequence=target_sequence,
                        calibration_id="lab-sim",
                        right_arm_q=latest.right_arm_q_rad,
                        right_arm_tau_ff=latest.right_arm_tau_ff_nm,
                        pipeline_age_ms=source_age_ms,
                    )
                    approach_completed |= "intercept_local_translation" in latest.reason

            previous_q = plant.state.arm_q[7:]
            previous_dq = plant.state.arm_dq[7:]
            controller.tick()
            assert controller.state is ArmState.ARMED, controller.state_report()
            assert plant.command is not None
            commanded_q = plant.command.q[7:]
            commanded_dq = plant.command.dq[7:]
            assert solver.validate_joint_path(
                (previous_q, commanded_q),
                support_plane=support,
                minimum_support_clearance_m=0.005,
                require_escape_cleared=False,
            ) is None
            maximum_speed = max(maximum_speed, max(abs(value) for value in commanded_dq))
            assert maximum_speed <= 1.0 + 1e-6
            acceleration = tuple(
                (current - previous) / dt
                for current, previous in zip(commanded_dq, previous_dq)
            )
            assert max(abs(value) for value in acceleration) <= 4.0 + 1e-4
            jerk = tuple(
                (current - previous) / dt
                for current, previous in zip(acceleration, previous_acceleration)
            )
            assert max(abs(value) for value in jerk) <= 30.0 + 1e-2
            previous_acceleration = acceleration
            plant.apply()
            clock.advance(dt)
            simulation_time_s += dt
            if approach_completed and max(abs(value) for value in commanded_dq) < 0.02:
                break
            time.sleep(dt)

        # The planner must reject an old state without poisoning the next update.
        client.submit(
            make_state(
                plant=plant,
                support=support,
                sequence=sequence - 1,
                simulation_time_s=simulation_time_s,
            )
        )
        stale = wait_for_command(client, sequence - 1, timeout_s=5.0)
        assert stale.status == "rejected"
        assert stale.reason == "stale_or_out_of_order_state"
        sequence += 1
        client.submit(
            make_state(
                plant=plant,
                support=support,
                sequence=sequence,
                simulation_time_s=simulation_time_s + 1.0 / 30.0,
            )
        )
        recovered = wait_for_command(client, sequence, timeout_s=5.0)
        assert recovered.reason != "stale_or_out_of_order_state"
    finally:
        client.close()

    assert approach_completed, reasons
    assert dropped_observations > 0
    assert controller.state is ArmState.ARMED
    assert controller.fault_reason is None
    assert controller.hold_reason is None
    assert plant.command_count >= int(0.95 * simulation_time_s * 250.0)
    assert maximum_speed > 0.20
    assert time.monotonic() - started_wall < 120.0
