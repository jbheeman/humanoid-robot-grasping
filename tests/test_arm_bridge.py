from __future__ import annotations

import math

import pytest

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
    ArmCommand,
    ArmState,
    RobotState,
)


class FakeClock:
    def __init__(self, monotonic: float = 10.0, wall: float = 1_800_000_000.0) -> None:
        self.monotonic = monotonic
        self.wall = wall

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += seconds


class FakeHardware:
    def __init__(self, state: RobotState | None) -> None:
        self.state = state
        self.commands: list[ArmCommand] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def latest_state(self) -> RobotState | None:
        return self.state

    def publish(self, command: ArmCommand) -> None:
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True


def robot_state(
    clock: FakeClock, q: tuple[float, ...] = (0.0,) * 14, **changes: object
) -> RobotState:
    values: dict[str, object] = {
        "arm_q": q,
        "received_at": clock.monotonic,
        "standing": True,
        "standing_since": clock.monotonic - 3.0,
        "compatible_motion_mode": True,
        "controller_available": True,
        "mode_machine": 5,
    }
    values.update(changes)
    return RobotState(**values)  # type: ignore[arg-type]


def bridge(clock: FakeClock, **config_changes: object) -> tuple[ArmBridgeController, FakeHardware]:
    hardware = FakeHardware(robot_state(clock))
    values: dict[str, object] = {"allow_movement": True, "calibration_id": "cal-1"}
    values.update(config_changes)
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(**values),  # type: ignore[arg-type]
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
    )
    return controller, hardware


def arm(controller: ArmBridgeController, hardware: FakeHardware, clock: FakeClock) -> None:
    starting_q = (0.0,) * 14 if hardware.state is None else hardware.state.arm_q
    controller.enable(session_id="session-a", calibration_id="cal-1")
    clock.advance(controller.config.startup_settle_s)
    hardware.state = robot_state(clock, q=starting_q)
    controller.tick()
    clock.advance(controller.config.weight_ramp_s)
    hardware.state = robot_state(clock, q=starting_q)
    controller.tick()
    assert controller.state is ArmState.ARMED


def test_movement_is_disabled_by_default_and_never_publishes() -> None:
    clock = FakeClock()
    hardware = FakeHardware(robot_state(clock))
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(calibration_id="cal-1"),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
    )

    with pytest.raises(ArmBridgeError, match="Movement is disabled") as rejected:
        controller.enable(session_id="session-a", calibration_id="cal-1")

    assert rejected.value.code == "movement_disabled"
    controller.tick()
    assert hardware.commands == []


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"standing": False, "standing_since": None}, "not_standing"),
        ({"standing_since": 9.5}, "standing_not_stable"),
        ({"compatible_motion_mode": False}, "incompatible_motion_mode"),
        ({"controller_available": False}, "competing_arm_controller"),
        ({"arm_q": (math.nan,) + (0.0,) * 13}, "robot_state_non_finite"),
    ],
)
def test_enable_requires_safe_stable_low_state(changes: dict[str, object], code: str) -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    hardware.state = robot_state(clock, **changes)

    with pytest.raises(ArmBridgeError) as rejected:
        controller.enable(session_id="session-a", calibration_id="cal-1")

    assert rejected.value.code == code
    assert controller.state is ArmState.DISARMED


def test_enable_requires_configured_matching_calibration() -> None:
    clock = FakeClock()
    controller, _ = bridge(clock, calibration_id=None)
    with pytest.raises(ArmBridgeError) as missing:
        controller.enable(session_id="session-a", calibration_id="cal-1")
    assert missing.value.code == "calibration_required"

    controller, _ = bridge(clock)
    with pytest.raises(ArmBridgeError) as mismatch:
        controller.enable(session_id="session-a", calibration_id="other")
    assert mismatch.value.code == "calibration_mismatch"


def test_verified_motion_mode_is_latched_for_the_active_tracking_session() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    period = 1.0 / controller.config.control_hz
    clock.advance(period)
    hardware.state = robot_state(clock, compatible_motion_mode=False)

    controller.tick()
    controller.set_target(
        session_id="session-a",
        sequence=0,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        source_timestamp=clock.wall,
    )

    assert controller.state is ArmState.ARMED
    assert controller.fault_reason is None


def test_arm_command_contains_all_joints_and_latches_left_arm() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    starting = tuple(index / 100.0 for index in range(14))
    hardware.state = robot_state(clock, q=starting)
    arm(controller, hardware, clock)

    target = [value + 0.04 for value in starting[7:]]
    controller.set_target(
        session_id="session-a",
        sequence=0,
        calibration_id="cal-1",
        right_arm_q=target,
        source_timestamp=clock.wall,
    )
    clock.advance(1.0 / controller.config.control_hz)
    hardware.state = robot_state(clock, q=starting)
    controller.tick()

    command = hardware.commands[-1]
    assert (
        len(command.q)
        == len(command.dq)
        == len(command.kp)
        == len(command.kd)
        == len(command.tau)
        == 14
    )
    assert command.q[:7] == starting[:7]
    assert command.mode_machine == 5
    assert command.kp == (80.0,) * 4 + (40.0,) * 3 + (80.0,) * 4 + (40.0,) * 3
    assert command.kd == (3.0,) * 4 + (1.5,) * 3 + (3.0,) * 4 + (1.5,) * 3
    assert 0.0 < command.q[7] - starting[7] < 0.04
    assert command.dq[:7] == (0.0,) * 7
    assert command.dq[7:] == pytest.approx(controller.right_velocity)
    assert command.dq[7] > 0.0
    assert command.weight == 1.0


def test_gravity_feedforward_is_validated_and_slew_limited() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    period = 1.0 / controller.config.control_hz
    requested_tau = (-2.0, 1.0, -0.5, -1.5, 0.2, -0.4, 0.1)

    controller.set_target(
        session_id="session-a",
        sequence=0,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        right_arm_tau_ff=requested_tau,
        source_timestamp=clock.wall,
    )
    clock.advance(period)
    hardware.state = robot_state(clock)
    controller.tick()

    maximum_step = controller.config.max_tau_ff_slew_nm_s * period
    assert max(abs(value) for value in controller.commanded_right_tau_ff) <= (
        maximum_step + 1e-9
    )
    assert hardware.commands[-1].tau[:7] == (0.0,) * 7
    assert hardware.commands[-1].tau[7:] == pytest.approx(
        controller.commanded_right_tau_ff
    )

    with pytest.raises(ArmBridgeError) as rejected:
        controller.set_target(
            session_id="session-a",
            sequence=1,
            calibration_id="cal-1",
            right_arm_q=[0.01] * 7,
            right_arm_tau_ff=[8.0] + [0.0] * 6,
            source_timestamp=clock.wall,
        )
    assert rejected.value.code == "torque_feedforward_limit"


def test_robot_local_gravity_uses_measured_state_on_every_250_hz_tick() -> None:
    class MeasuredGravity:
        def __init__(self) -> None:
            self.inputs: list[tuple[float, ...]] = []

        def torque(self, q: tuple[float, ...]) -> tuple[float, ...]:
            self.inputs.append(tuple(q))
            return (-1.0, 0.0, 0.0, -3.5 - q[3], 0.0, 0.0, 0.0)

    clock = FakeClock()
    hardware = FakeHardware(robot_state(clock))
    gravity = MeasuredGravity()
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            allow_movement=True,
            calibration_id="cal-1",
            max_tau_ff_slew_nm_s=1_000.0,
            max_following_error_rad=1.0,
        ),
        monotonic=lambda: clock.monotonic,
        wall_time=lambda: clock.wall,
        gravity_compensator=gravity,
    )
    arm(controller, hardware, clock)
    initial_updates = controller.metrics.gravity_updates
    period = 1.0 / controller.config.control_hz

    for elbow in (0.2, 0.4, 0.6):
        measured = (0.0,) * 7 + (0.0, 0.0, 0.0, elbow, 0.0, 0.0, 0.0)
        clock.advance(period)
        hardware.state = robot_state(clock, q=measured)
        controller.tick()

    assert gravity.inputs[-3:] == [
        (0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0),
    ]
    assert controller.metrics.gravity_updates == initial_updates + 3
    assert hardware.commands[-1].tau[10] == pytest.approx(-4.1)
    health = controller.health_report()
    assert health["gravity_feedforward_source"] == "measured_state_urdf"
    assert health["gravity_update_hz"] == 250.0


def test_ruckig_limits_velocity_acceleration_and_jerk_during_replanning() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    period = 1.0 / controller.config.control_hz
    controller.set_target(
        session_id="session-a",
        sequence=0,
        calibration_id="cal-1",
        right_arm_q=[0.04] * 7,
        source_timestamp=clock.wall,
    )

    previous_acceleration = list(controller.right_acceleration)
    for _ in range(30):
        clock.advance(period)
        hardware.state = robot_state(clock)
        controller.tick()
        assert max(abs(value) for value in controller.right_velocity) <= (
            controller.config.max_velocity_rad_s + 1e-9
        )
        assert max(abs(value) for value in controller.right_acceleration) <= (
            controller.config.max_acceleration_rad_s2 + 1e-9
        )
        jerk = [
            (actual - previous) / period
            for actual, previous in zip(
                controller.right_acceleration,
                previous_acceleration,
            )
        ]
        assert max(abs(value) for value in jerk) <= controller.config.max_jerk_rad_s3 + 1e-6
        previous_acceleration = list(controller.right_acceleration)

    assert controller.commanded_right is not None
    replacement = [value - 0.03 for value in controller.commanded_right]
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=replacement,
        source_timestamp=clock.wall,
    )
    clock.advance(period)
    hardware.state = robot_state(clock)
    controller.tick()
    replanned_jerk = [
        (actual - previous) / period
        for actual, previous in zip(
            controller.right_acceleration,
            previous_acceleration,
        )
    ]
    assert max(abs(value) for value in replanned_jerk) <= (
        controller.config.max_jerk_rad_s3 + 1e-6
    )


def test_ruckig_online_output_persists_across_control_ticks() -> None:
    clock = FakeClock()
    starting = (0.0,) * 7 + (0.28, -0.13, 0.01, 0.98, -0.13, 0.02, -0.01)
    controller, hardware = bridge(
        clock,
        weight_ramp_s=1.5,
        max_velocity_rad_s=0.15,
        max_acceleration_rad_s2=0.5,
        max_jerk_rad_s3=2.0,
    )
    hardware.state = robot_state(clock, q=starting)
    target = [0.2675, -0.165, 0.01, 0.98, -0.13, 0.02, -0.01]
    controller.enable(session_id="session-a", calibration_id="cal-1")
    controller.set_target(
        session_id="session-a",
        sequence=0,
        calibration_id="cal-1",
        right_arm_q=target,
        source_timestamp=clock.wall,
    )
    clock.advance(controller.config.startup_settle_s)
    hardware.state = robot_state(clock, q=starting)
    controller.tick()
    clock.advance(controller.config.weight_ramp_s)
    hardware.state = robot_state(clock, q=starting)
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=target,
        source_timestamp=clock.wall,
    )
    controller.tick()
    assert controller.state is ArmState.ARMED
    controller.set_target(
        session_id="session-a",
        sequence=2,
        calibration_id="cal-1",
        right_arm_q=target,
        source_timestamp=clock.wall,
    )
    period = 1.0 / controller.config.control_hz

    positions = []
    for _ in range(8):
        clock.advance(period)
        hardware.state = robot_state(clock, q=starting)
        controller.tick()
        assert controller.commanded_right is not None
        positions.append(tuple(controller.commanded_right))

    # Recreating OutputParameter on every update makes Ruckig return a
    # zero-filled Finished result on the second tick. The live robot then
    # receives a near-full-weight command toward zero and jerks.
    assert all(position[3] > 0.97 for position in positions)
    assert all(position != (0.0,) * 7 for position in positions)
    assert positions[-1][1] < positions[0][1]


def test_target_rejects_replay_wrong_binding_stale_nonfinite_and_discontinuity() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    valid = [0.01] * 7

    def submit(**changes: object) -> None:
        values: dict[str, object] = {
            "session_id": "session-a",
            "sequence": 1,
            "calibration_id": "cal-1",
            "right_arm_q": valid,
            "source_timestamp": clock.wall,
        }
        values.update(changes)
        controller.set_target(**values)

    submit()
    cases = [
        ({"sequence": 1}, "stale_sequence"),
        ({"sequence": 2, "session_id": "wrong"}, "session_mismatch"),
        ({"sequence": 2, "calibration_id": "wrong"}, "calibration_mismatch"),
        ({"sequence": 2, "source_timestamp": clock.wall - 0.501}, "stale_target"),
        ({"sequence": 2, "right_arm_q": [math.nan] * 7}, "invalid_target"),
    ]
    for changes, expected_code in cases:
        with pytest.raises(ArmBridgeError) as rejected:
            submit(**changes)
        assert rejected.value.code == expected_code


def test_joint_limits_include_margin() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    target = [0.0] * 7
    target[0] = controller.config.right_joint_limits[0][1] - 0.049

    with pytest.raises(ArmBridgeError) as rejected:
        controller.set_target(
            session_id="session-a",
            sequence=1,
            calibration_id="cal-1",
            right_arm_q=target,
            source_timestamp=clock.wall,
        )

    assert rejected.value.code == "joint_limit"


def test_deadman_holds_and_releases_weight_within_bounded_ramp() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        source_timestamp=clock.wall,
    )

    clock.advance(controller.config.deadman_s + 0.001)
    hardware.state = robot_state(clock)
    controller.tick()
    assert controller.state is ArmState.HOLDING
    assert controller.hold_reason == "target_deadman"
    assert hardware.commands[-1].weight == 1.0

    clock.advance(controller.config.weight_ramp_s)
    hardware.state = robot_state(clock)
    controller.tick()
    assert hardware.commands[-1].weight == 0.0
    assert controller.state is ArmState.DISARMED


def test_tracking_target_during_arming_keeps_deadman_alive() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    controller.start()
    controller.enable(session_id="session-a", calibration_id="cal-1")

    assert controller.state is ArmState.ARMING
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        source_timestamp=clock.wall,
    )
    clock.advance(controller.config.startup_settle_s)
    hardware.state = robot_state(clock)
    controller.tick()
    clock.advance(controller.config.weight_ramp_s)
    hardware.state = robot_state(clock)
    controller.set_target(
        session_id="session-a",
        sequence=2,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        source_timestamp=clock.wall,
    )
    controller.tick()

    assert controller.state is ArmState.ARMED
    assert controller.hold_reason is None


def test_tracking_target_during_arming_is_not_released_after_ramp() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    controller.enable(session_id="session-a", calibration_id="cal-1")
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=[0.04] * 7,
        source_timestamp=clock.wall,
    )

    clock.advance(controller.config.startup_settle_s)
    hardware.state = robot_state(clock)
    controller.tick()
    clock.advance(controller.config.weight_ramp_s)
    hardware.state = robot_state(clock)
    controller.tick()

    assert controller.state is ArmState.ARMED
    assert controller.commanded_right == [0.0] * 7
    assert controller.desired_right == (0.0,) * 7
    assert controller.state_report()["loop"]["deferred_arming_targets"] == 1  # type: ignore[index]


def test_tracking_heartbeat_does_not_extend_target_deadman() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=[0.01] * 7,
        source_timestamp=clock.wall,
    )

    clock.advance(controller.config.deadman_s - 0.001)
    hardware.state = robot_state(clock)
    assert controller.heartbeat(session_id="session-a")["state"] == "ARMED"
    clock.advance(0.002)
    hardware.state = robot_state(clock)
    controller.tick()

    assert controller.state is ArmState.HOLDING
    assert controller.hold_reason == "target_deadman"


def test_stale_robot_state_faults_and_ramps_weight_to_zero() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    clock.advance(controller.config.state_ttl_s + 0.001)
    controller.tick()
    assert controller.state is ArmState.FAULT
    assert controller.fault_reason == "robot_state_stale"

    clock.advance(controller.config.weight_ramp_s)
    controller.tick()
    assert controller.state is ArmState.FAULT
    assert hardware.commands[-1].weight == 0.0


def test_tracking_relies_on_robot_balance_controller_after_enable() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    controller.set_target(
        session_id="session-a",
        sequence=1,
        calibration_id="cal-1",
        right_arm_q=[0.5] * 7,
        source_timestamp=clock.wall,
    )

    clock.advance(1.0 / controller.config.control_hz)
    hardware.state = robot_state(
        clock,
        standing=False,
        standing_since=None,
        balance_details=("imu_level",),
        waist_q=(0.5, 0.5, 0.5),
    )
    controller.tick()

    assert controller.state is ArmState.ARMED
    assert controller.commanded_right is not None
    assert max(controller.commanded_right) > 0.0


def test_stop_is_idempotent_and_high_priority() -> None:
    clock = FakeClock()
    controller, hardware = bridge(clock)
    arm(controller, hardware, clock)
    first = controller.stop()
    second = controller.stop()
    assert first["state"] == second["state"] == "HOLDING"
    assert controller.hold_reason == "operator_stop"


def test_rate_is_verified_250_hz_default_and_reports_periods() -> None:
    config = ArmBridgeConfig()
    assert config.control_hz == 250.0
    assert config.max_jerk_rad_s3 == 20.0
    with pytest.raises(ValueError, match="50-250 Hz"):
        ArmBridgeConfig(control_hz=251.0)
    with pytest.raises(ValueError, match="50-250 Hz"):
        ArmBridgeConfig(control_hz=49.0)
    with pytest.raises(ValueError, match="max_jerk_rad_s3"):
        ArmBridgeConfig(max_jerk_rad_s3=0.0)

    clock = FakeClock()
    controller, _ = bridge(clock)
    controller.tick()
    clock.advance(1.0 / 250.0)
    controller.tick()
    assert controller.health_report()["loop"]["last_period_ms"] == 4.0  # type: ignore[index]
