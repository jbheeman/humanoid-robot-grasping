from __future__ import annotations

import sys
from types import ModuleType

from scripts.robot.arm_ros_relay import ArmRosRelay


class Message:
    def __init__(self) -> None:
        self.data = ""


class Policy:
    KEEP_LAST = 1
    RELIABLE = 2
    VOLATILE = 3


class FakePublisher:
    def publish(self, message: object) -> None:
        del message


class FakeNode:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def create_publisher(
        self, message_type: type, topic: str, qos: object
    ) -> FakePublisher:
        del message_type, qos
        self.events.append(f"publisher:{topic}")
        return FakePublisher()

    def create_subscription(
        self, message_type: type, topic: str, callback: object, qos: object
    ) -> object:
        del message_type, callback, qos
        self.events.append(f"subscription:{topic}")
        return object()

    def create_timer(self, period_s: float, callback: object) -> object:
        del callback
        assert period_s == 0.05
        self.events.append("timer")
        return object()


class FakeRunner:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.node = FakeNode(self.events)

    def prepare(self) -> FakeNode:
        self.events.append("prepare")
        return self.node

    def start(self) -> FakeNode:
        self.events.append("start")
        return self.node

    def close(self) -> None:
        self.events.append("close")


class FakeClient:
    def call(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del operation, payload
        return {"ok": True, "report": {}}


def test_relay_creates_all_entities_before_executor_starts(monkeypatch: object) -> None:
    qos = ModuleType("rclpy.qos")
    qos.DurabilityPolicy = Policy  # type: ignore[attr-defined]
    qos.HistoryPolicy = Policy  # type: ignore[attr-defined]
    qos.ReliabilityPolicy = Policy  # type: ignore[attr-defined]
    qos.QoSProfile = lambda **values: values  # type: ignore[attr-defined]
    rclpy = ModuleType("rclpy")
    rclpy.qos = qos  # type: ignore[attr-defined]
    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.String = Message  # type: ignore[attr-defined]
    std_msgs.msg = std_msgs_msg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)  # type: ignore[attr-defined]
    runner = FakeRunner()

    relay = ArmRosRelay(
        "/unused/test.sock",
        runner=runner,  # type: ignore[arg-type]
        client=FakeClient(),  # type: ignore[arg-type]
    )

    assert runner.events == [
        "prepare",
        "publisher:/g1/arm/state_json",
        "publisher:/g1/arm/control/response_json",
        "subscription:/g1/arm/target_json",
        "subscription:/g1/arm/control/request_json",
        "timer",
        "start",
    ]
    relay.close()
