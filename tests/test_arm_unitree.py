from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

from object_tracking.arm_tracking.arm_bridge import ArmCommand
from object_tracking.arm_tracking.arm_unitree import (
    ARM_COMMAND_TOPIC,
    ARM_INDICES,
    ARM_WEIGHT_INDEX,
    LOW_STATE_TOPIC,
    MOTION_REQUEST_TOPIC,
    MOTION_RESPONSE_TOPIC,
    UnitreeArmHardware,
)
from object_tracking.ros2_transport import Ros2Bindings


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


class Request:
    def __init__(self) -> None:
        self.header = SimpleNamespace(identity=SimpleNamespace(id=0, api_id=0))
        self.parameter = ""
        self.binary = []


class Response:
    def __init__(self, *, request_id: int, data: str, status: int = 0) -> None:
        self.header = SimpleNamespace(
            identity=SimpleNamespace(id=request_id, api_id=1001),
            status=SimpleNamespace(code=status),
        )
        self.data = data
        self.binary = []


class Publisher:
    def __init__(self, node: "Node", topic: str) -> None:
        self.node = node
        self.topic = topic
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)
        if self.topic == MOTION_REQUEST_TOPIC:
            response = Response(
                request_id=message.header.identity.id,
                data=json.dumps({"form": "0", "name": "ai"}),
            )
            self.node.emit(MOTION_RESPONSE_TOPIC, response)
        elif self.topic == ARM_COMMAND_TOPIC:
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


class Runner:
    def __init__(self, node: Node, bindings: Ros2Bindings) -> None:
        self.node = node
        self.bindings = bindings
        self.started = False
        self.closed = False

    def start(self) -> Node:
        self.started = True
        return self.node

    def close(self) -> None:
        self.closed = True


def bindings() -> Ros2Bindings:
    return Ros2Bindings(
        rclpy=SimpleNamespace(),
        context_type=object,
        executor_type=object,
        low_cmd_type=LowCommand,
        low_state_type=LowState,
        request_type=Request,
        response_type=Response,
    )


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
    command = arm_command()

    hardware.publish(command)

    assert publisher.messages == [low_command]
    assert low_command.mode_pr == 0
    assert low_command.mode_machine == 5
    assert low_command.crc == 0
    for offset, joint_index in enumerate(ARM_INDICES):
        motor = low_command.motor_cmd[joint_index]
        assert motor.q == command.q[offset]
        assert motor.dq == 0.0
        assert motor.kp == 60.0
        assert motor.kd == 1.5
        assert motor.tau == 0.0
    assert low_command.motor_cmd[ARM_WEIGHT_INDEX].q == 0.75
    assert [low_command.motor_cmd[index].q for index in (12, 13, 14)] == [-999.0] * 3
    assert [low_command.motor_cmd[index].q for index in range(12)] == [-999.0] * 12


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


def test_start_wires_official_topics_and_verifies_mode_and_ownership() -> None:
    now = [10.0]
    node = Node()
    runner = Runner(node, bindings())
    hardware = UnitreeArmHardware(
        expected_motion_mode="ai",
        ownership_quiet_s=0.0,
        motion_poll_s=0.01,
        rpc_timeout_s=0.1,
        monotonic=lambda: now[0],
        ros_runner=runner,
    )

    hardware.start()
    node.emit(LOW_STATE_TOPIC, LowState())
    for _ in range(100):
        state = hardware.latest_state()
        if state is not None and state.motion_mode_verified:
            break
        hardware._motion_stop.wait(0.001)

    assert runner.started is True
    assert set(node.publishers) == {ARM_COMMAND_TOPIC, MOTION_REQUEST_TOPIC}
    assert set(node.subscriptions) == {
        LOW_STATE_TOPIC,
        ARM_COMMAND_TOPIC,
        MOTION_RESPONSE_TOPIC,
    }
    assert state is not None
    assert state.motion_mode_name == "ai"
    assert state.compatible_motion_mode is True
    assert state.controller_ownership_verified is True
    assert state.controller_available is True

    node.arm_publisher_count = 2
    state = hardware.latest_state()
    assert state is not None
    assert state.controller_available is False
    assert state.controller_ownership_verified is False

    hardware.close()
    # An injected runner is shared and remains the caller's responsibility.
    assert runner.closed is False


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
