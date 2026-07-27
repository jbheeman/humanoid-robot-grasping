from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from object_tracking.arm_tracking.arm_bridge import ArmCommand
from object_tracking.arm_tracking.arm_unitree import (
    ARM_COMMAND_TOPIC,
    ARM_INDICES,
    ARM_WEIGHT_INDEX,
    UnitreeArmHardware,
)


@dataclass
class Motor:
    mode: int = 0
    q: float = -999.0
    dq: float = -999.0
    kp: float = -999.0
    kd: float = -999.0
    tau: float = -999.0


class LowCommand:
    def __init__(self) -> None:
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [Motor() for _ in range(35)]
        self.crc = 0


class StateMotor:
    def __init__(self, q: float = 0.0, dq: float = 0.0) -> None:
        self.q = q
        self.dq = dq
        # Unitree ROS 2 v0.3.0 defines MotorState.temperature as int16[2].
        self.temperature = [30, 31]
        self.motorstate = 0


class Imu:
    rpy = (0.0, 0.0, 0.0)
    gyroscope = (0.0, 0.0, 0.0)


class LowState:
    def __init__(self) -> None:
        self.motor_state = [StateMotor() for _ in range(35)]
        self.imu_state = Imu()
        self.mode_machine = 5


class Publisher:
    def __init__(self, node: "Node", topic: str) -> None:
        self.node = node
        self.topic = topic
        self.messages: list[object] = []

    def Write(self, message: object) -> None:
        self.messages.append(message)
        if self.topic == ARM_COMMAND_TOPIC:
            self.node.emit(ARM_COMMAND_TOPIC, message)


class Node:
    def __init__(self) -> None:
        self.publishers: dict[str, Publisher] = {}
        self.subscriptions: dict[str, list[object]] = {}
        self.arm_publisher_count = 1
        self.destroyed: list[object] = []

    def create_publisher(self, message_type: type, topic: str, qos_depth: int) -> Publisher:
        del message_type, qos_depth
        publisher = Publisher(self, topic)
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(
        self,
        message_type: type,
        topic: str,
        callback: object,
        qos_depth: int,
    ) -> object:
        del message_type, qos_depth
        subscription = SimpleNamespace(topic=topic, callback=callback)
        self.subscriptions.setdefault(topic, []).append(subscription)
        return subscription

    def emit(self, topic: str, message: object) -> None:
        for subscription in tuple(self.subscriptions.get(topic, ())):
            subscription.callback(message)

    def count_publishers(self, topic: str) -> int:
        return self.arm_publisher_count if topic == ARM_COMMAND_TOPIC else 0

    def destroy_subscription(self, subscription: object) -> None:
        self.destroyed.append(subscription)

    def destroy_publisher(self, publisher: object) -> None:
        self.destroyed.append(publisher)


def arm_command() -> ArmCommand:
    return ArmCommand(
        q=tuple(float(index) / 10.0 for index in range(14)),
        dq=(0.0,) * 14,
        kp=(60.0,) * 14,
        kd=(1.5,) * 14,
        weight=0.75,
        mode_machine=5,
        published_at=10.0,
    )


def test_ros_adapter_writes_all_fourteen_arm_slots_and_weight_without_crc() -> None:
    hardware = UnitreeArmHardware()
    low_command = LowCommand()
    node = Node()
    publisher = Publisher(node, ARM_COMMAND_TOPIC)
    hardware._started = True
    hardware._low_cmd = low_command
    hardware._publisher = publisher
    hardware._crc = SimpleNamespace(Crc=lambda message: 123)
    hardware._low_state_callback(LowState())
    command = arm_command()

    hardware.publish(command)

    assert publisher.messages == [low_command]
    assert low_command.mode_pr == 0
    assert low_command.mode_machine == 5
    assert low_command.crc == 123
    for offset, joint_index in enumerate(ARM_INDICES):
        motor = low_command.motor_cmd[joint_index]
        assert motor.q == command.q[offset]
        assert motor.dq == 0.0
        assert motor.kp == 60.0
        assert motor.kd == 1.5
        assert motor.tau == 0.0
    assert low_command.motor_cmd[ARM_WEIGHT_INDEX].q == 0.75
    assert [low_command.motor_cmd[index].q for index in range(15)] == [0.0] * 15
    assert low_command.motor_cmd[0].kp == 300.0
    assert low_command.motor_cmd[4].kp == 80.0
    assert low_command.motor_cmd[12].kp == 300.0


def test_standing_excludes_arm_velocity_and_reads_ros_temperature_arrays() -> None:
    hardware = UnitreeArmHardware(monotonic=lambda: 10.0)
    message = LowState()
    for index in ARM_INDICES:
        message.motor_state[index].dq = 0.4

    hardware._low_state_callback(message)

    assert hardware._state is not None
    assert hardware._state.standing is True
    assert hardware._state.arm_dq == (0.4,) * 14
    assert hardware._state.motor_status_verified is True
    assert hardware._state.motor_state_healthy is True

    message.motor_state[7].temperature[1] = 90
    hardware._low_state_callback(message)
    assert hardware._state is not None
    assert hardware._state.motor_state_healthy is False
    assert hardware._state.motor_faults == ("motor_7_temperature_1",)

    message.motor_state[0].dq = 0.3
    hardware._low_state_callback(message)
    assert hardware._state is not None
    assert hardware._state.standing is False
    assert "legs_waist_still" in hardware._state.balance_details


def test_lowstate_captures_all_29_encoder_positions_and_velocities() -> None:
    hardware = UnitreeArmHardware(monotonic=lambda: 10.0)
    message = LowState()
    for index in range(29):
        message.motor_state[index].q = index / 10.0
        message.motor_state[index].dq = index / 100.0

    hardware._low_state_callback(message)

    assert hardware._state is not None
    assert hardware._state.body_q == tuple(index / 10.0 for index in range(29))
    assert hardware._state.body_dq == tuple(index / 100.0 for index in range(29))
    assert hardware._state.arm_q == tuple(index / 10.0 for index in ARM_INDICES)
    assert hardware._state.arm_dq == tuple(index / 100.0 for index in ARM_INDICES)


@pytest.mark.skip(reason="requires the physical robot's native Unitree SDK2 DDS")
def test_native_sdk_start_is_integration_covered() -> None:
    """Native DDS decoding is validated read-only on the lab G1."""


def test_foreign_arm_message_latches_ownership_conflict() -> None:
    hardware = UnitreeArmHardware(monotonic=lambda: 5.0, ownership_quiet_s=0.0)
    node = Node()
    hardware._node = node
    hardware._started_at = 5.0
    hardware._motion_mode_checked = True
    hardware._motion_mode_name = "ai"
    hardware.expected_motion_mode = "ai"
    hardware._low_state_callback(LowState())

    hardware._arm_command_callback(LowCommand())

    state = hardware.latest_state()
    assert state is not None
    assert state.controller_available is False
    assert state.controller_ownership_verified is False


def test_motion_mode_poll_keeps_a_recent_verified_mode_through_one_rpc_failure() -> None:
    now = [10.0]
    hardware = UnitreeArmHardware(
        monotonic=lambda: now[0], motion_poll_s=0.01, motion_mode_grace_s=5.0
    )

    class Client:
        calls = 0

        def CheckMode(self) -> object:
            self.calls += 1
            if self.calls == 1:
                return (0, {"name": "ai"})
            raise RuntimeError("temporary rpc timeout")

    class StopAfterTwoPolls:
        def __init__(self) -> None:
            self.stopped = False
            self.waits = 0

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, _: float) -> bool:
            self.waits += 1
            if self.waits >= 2:
                self.stopped = True
            return self.stopped

    hardware._motion_client = Client()
    hardware._motion_stop = StopAfterTwoPolls()  # type: ignore[assignment]
    hardware._poll_motion_mode()

    assert hardware._motion_mode_name == "ai"
    assert hardware._motion_mode_checked is True
    assert hardware._motion_mode_verified_at == 10.0
    assert hardware._motion_mode_error == "RuntimeError: temporary rpc timeout"


def test_motion_mode_verification_expires_after_bounded_grace() -> None:
    now = [10.0]
    hardware = UnitreeArmHardware(monotonic=lambda: now[0], motion_mode_grace_s=5.0)
    hardware.expected_motion_mode = "ai"
    hardware._motion_mode_name = "ai"
    hardware._motion_mode_checked = True
    hardware._motion_mode_verified_at = 10.0
    hardware._low_state_callback(LowState())

    assert hardware.latest_state() is not None
    assert hardware.latest_state().compatible_motion_mode is True
    now[0] = 15.01
    assert hardware.latest_state() is not None
    assert hardware.latest_state().compatible_motion_mode is False


def test_default_motion_mode_grace_tolerates_stock_rpc_stalls() -> None:
    hardware = UnitreeArmHardware()
    assert hardware.motion_mode_grace_s == 30.0
