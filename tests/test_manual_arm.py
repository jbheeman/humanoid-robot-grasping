from __future__ import annotations

import math

import pytest

from object_tracking.arm_tracking.arm_bridge import ArmBridgeError, ArmCommand, ArmState, RobotState
from object_tracking.arm_tracking.joints import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from object_tracking.arm_tracking.manual_arm import ManualArmConfig, ManualArmController


class Clock:
    def __init__(self) -> None:
        self.monotonic = 10.0
        self.wall_ns = 1_800_000_000_000_000_000

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall_ns += int(seconds * 1e9)


class Hardware:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.state = self.make_state()
        self.commands: list[ArmCommand] = []
        self.started = False
        self.closed = False

    def make_state(self, q: tuple[float, ...] = (0.0,) * 14, **changes: object) -> RobotState:
        values: dict[str, object] = {
            "arm_q": q,
            "arm_dq": (0.0,) * 14,
            "received_at": self.clock.monotonic,
            "standing": True,
            "standing_since": self.clock.monotonic - 3.0,
            "compatible_motion_mode": True,
            "controller_available": True,
            "mode_machine": 5,
            "motion_mode_name": "ai",
            "motion_mode_verified": True,
            "controller_ownership_verified": True,
            "motor_status_verified": True,
            "motor_state_healthy": True,
        }
        values.update(changes)
        return RobotState(**values)  # type: ignore[arg-type]

    def refresh(self, q: tuple[float, ...] | None = None) -> None:
        self.state = self.make_state(self.state.arm_q if q is None else q)

    def start(self) -> None:
        self.started = True

    def latest_state(self) -> RobotState:
        return self.state

    def publish(self, command: ArmCommand) -> None:
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True


def setup() -> tuple[Clock, Hardware, ManualArmController]:
    clock = Clock()
    hardware = Hardware(clock)
    controller = ManualArmController(
        hardware,
        ManualArmConfig(allow_movement=True, weight_ramp_s=0.5),
        monotonic=lambda: clock.monotonic,
        wall_time_ns=lambda: clock.wall_ns,
    )
    return clock, hardware, controller


def arm(clock: Clock, hardware: Hardware, controller: ManualArmController) -> str:
    report = controller.enable("session-a")
    assert report["state"] == ArmState.ARMING.value
    clock.advance(0.5)
    hardware.refresh()
    controller.heartbeat("session-a")
    controller.tick()
    assert controller.state_report()["state"] == ArmState.ARMED.value
    return "session-a"


def target(
    controller: ManualArmController,
    clock: Clock,
    *,
    side: str,
    sequence: int,
    positions: tuple[float, ...],
    duration: float = 1.0,
) -> None:
    controller.set_side_target(
        side=side,
        session_id="session-a",
        sequence=sequence,
        joint_names=LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES,
        position_rad=positions,
        duration_s=duration,
        sent_time_ns=clock.wall_ns,
    )


def test_left_and_right_trajectories_are_independent_and_publish_full_frame() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    target(controller, clock, side="left", sequence=0, positions=(0.05,) + (0.0,) * 6)
    clock.advance(0.25)
    hardware.refresh()
    controller.heartbeat("session-a")
    controller.tick()
    left_at_quarter = hardware.commands[-1].q[0]
    assert 0.0 < left_at_quarter < 0.05
    assert hardware.commands[-1].q[7:] == (0.0,) * 7

    target(controller, clock, side="right", sequence=0, positions=(0.05,) + (0.0,) * 6)
    clock.advance(0.25)
    hardware.refresh()
    controller.heartbeat("session-a")
    controller.tick()
    command = hardware.commands[-1]
    assert len(command.q) == 14
    assert command.q[0] > left_at_quarter
    assert 0.0 < command.q[7] < 0.05


def test_sequences_are_per_side_and_replay_is_rejected() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    target(controller, clock, side="left", sequence=0, positions=(0.01,) + (0.0,) * 6)
    target(controller, clock, side="right", sequence=0, positions=(0.01,) + (0.0,) * 6)
    with pytest.raises(ArmBridgeError, match="sequence"):
        target(controller, clock, side="left", sequence=0, positions=(0.02,) + (0.0,) * 6)


def test_arming_rebases_commands_to_the_measured_settled_pose() -> None:
    clock, hardware, controller = setup()
    controller.enable("session-a")
    settled = tuple(0.01 * (index + 1) for index in range(14))
    clock.advance(0.5)
    hardware.state = hardware.make_state(q=settled)
    controller.heartbeat("session-a")
    controller.tick()

    report = controller.state_report()
    assert report["state"] == ArmState.ARMED.value
    assert report["baseline_arm_q"] == list(settled)
    assert report["commanded_arm_q"] == list(settled)
    assert report["desired_arm_q"] == list(settled)
    assert hardware.commands[-1].q == settled


@pytest.mark.parametrize(
    ("names", "positions", "match"),
    [
        (RIGHT_ARM_JOINT_NAMES[::-1], (0.0,) * 7, "canonical"),
        (RIGHT_ARM_JOINT_NAMES, (math.nan,) + (0.0,) * 6, "finite"),
    ],
)
def test_target_contract_rejections(names, positions, match) -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    with pytest.raises(ArmBridgeError, match=match):
        controller.set_side_target(
            side="right",
            session_id="session-a",
            sequence=0,
            joint_names=names,
            position_rad=positions,
            duration_s=1.0,
            sent_time_ns=clock.wall_ns,
        )


def test_oversized_target_is_clamped_instead_of_rejected() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    report = controller.set_side_target(
        side="right",
        session_id="session-a",
        sequence=0,
        joint_names=RIGHT_ARM_JOINT_NAMES,
        position_rad=(0.20,) + (0.0,) * 6,
        duration_s=1.0,
        sent_time_ns=clock.wall_ns,
    )
    assert report["last_rejection"] is None
    assert report["last_clamp"] == {
        "side": "right",
        "maximum_requested_delta_rad": pytest.approx(0.20),
        "maximum_applied_delta_rad": pytest.approx(0.05),
    }
    assert report["desired_arm_q"][7] == pytest.approx(0.05)


def test_heartbeat_timeout_ramps_to_zero_and_stop_resets_fault() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    clock.advance(0.501)
    hardware.refresh()
    controller.tick()
    assert controller.state_report()["state"] == ArmState.HOLDING.value
    assert hardware.commands[-1].weight == pytest.approx(1.0)
    clock.advance(0.5)
    hardware.refresh()
    controller.tick()
    assert controller.state_report()["state"] == ArmState.DISARMED.value
    assert controller.state_report()["weight"] == 0.0

    arm(clock, hardware, controller)
    hardware.state = hardware.make_state(motion_mode_verified=False)
    controller.tick()
    clock.advance(0.5)
    hardware.refresh()
    controller.tick()
    assert controller.state_report()["state"] == ArmState.FAULT.value
    assert controller.stop("operator_reset")["state"] == ArmState.DISARMED.value


def test_disabled_bridge_and_stale_timestamp_fail_closed() -> None:
    clock, hardware, _controller = setup()
    disabled = ManualArmController(
        hardware,
        ManualArmConfig(allow_movement=False),
        monotonic=lambda: clock.monotonic,
        wall_time_ns=lambda: clock.wall_ns,
    )
    with pytest.raises(ArmBridgeError, match="disabled"):
        disabled.enable()

    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    with pytest.raises(ArmBridgeError, match="not fresh"):
        controller.set_side_target(
            side="right",
            session_id="session-a",
            sequence=0,
            joint_names=RIGHT_ARM_JOINT_NAMES,
            position_rad=(0.01,) + (0.0,) * 6,
            duration_s=1.0,
            sent_time_ns=clock.wall_ns - 1_000_000_000,
        )


def test_manual_commands_use_installed_sdk2_example_gains() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    command = hardware.commands[-1]
    assert command.kp == (60.0,) * 14
    assert command.kd == (1.5,) * 14


def test_velocity_gate_allows_guarded_motion_and_names_excessive_joint() -> None:
    clock, hardware, controller = setup()
    hardware.state = hardware.make_state(arm_dq=(0.30,) + (0.0,) * 13)
    assert controller._gate_failures(hardware.state, clock.monotonic) == []

    hardware.state = hardware.make_state(
        arm_dq=(0.0,) * 12 + (1.01, 0.0),
    )
    assert controller._gate_failures(hardware.state, clock.monotonic) == [
        "arm_velocity_too_high:right_wrist_pitch_joint:1.0100"
    ]


def test_following_error_fault_names_the_diverging_joint() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    controller._commanded_q = (0.151,) + (0.0,) * 13
    controller.tick()
    assert controller.state_report()["fault_reason"] == (
        "following_error:left_shoulder_pitch_joint:0.1510"
    )


def test_release_does_not_latch_a_following_error() -> None:
    clock, hardware, controller = setup()
    arm(clock, hardware, controller)
    controller._commanded_q = (0.20,) + (0.0,) * 13
    controller.stop("operator_stop")
    controller.tick()
    assert controller.state_report()["state"] == ArmState.HOLDING.value
    assert controller.state_report()["fault_reason"] is None


def test_manual_command_gains_follow_configuration() -> None:
    clock, hardware, _controller = setup()
    controller = ManualArmController(
        hardware,
        ManualArmConfig(allow_movement=True, kp=55.0, kd=1.25),
        monotonic=lambda: clock.monotonic,
        wall_time_ns=lambda: clock.wall_ns,
    )
    arm(clock, hardware, controller)
    command = hardware.commands[-1]
    assert command.kp == (55.0,) * 14
    assert command.kd == (1.25,) * 14


def test_manual_xr_profile_uses_unitree_xr_gains() -> None:
    clock, hardware, _controller = setup()
    controller = ManualArmController(
        hardware,
        ManualArmConfig(allow_movement=True, gain_profile="xr"),
        monotonic=lambda: clock.monotonic,
        wall_time_ns=lambda: clock.wall_ns,
    )
    arm(clock, hardware, controller)
    command = hardware.commands[-1]
    assert command.kp == (80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0) * 2
    assert command.kd == (3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5) * 2


def test_manual_controller_emits_sparse_diagnostic_events() -> None:
    clock = Clock()
    hardware = Hardware(clock)
    events: list[dict[str, object]] = []
    controller = ManualArmController(
        hardware,
        ManualArmConfig(allow_movement=True, weight_ramp_s=0.5),
        monotonic=lambda: clock.monotonic,
        wall_time_ns=lambda: clock.wall_ns,
        event_sink=events.append,
    )

    controller.enable("session-a")
    clock.advance(0.5)
    hardware.refresh()
    controller.heartbeat("session-a")
    controller.tick()
    target(
        controller,
        clock,
        side="right",
        sequence=0,
        positions=(0.01,) + (0.0,) * 6,
    )
    controller.stop("test_complete")

    names = [str(event["event"]) for event in events]
    assert names == [
        "enable_accepted",
        "armed",
        "target_accepted",
        "release_started",
    ]
    accepted = events[2]
    assert accepted["side"] == "right"
    assert accepted["sequence"] == 0
    assert accepted["target_q"] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
