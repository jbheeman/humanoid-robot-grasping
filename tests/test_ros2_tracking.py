from __future__ import annotations

import json
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from object_tracking.arm_tracking.protocol import DepthEnvelopeCodec
from object_tracking.ros2_tracking import RosTrackingError, RosTrackingTransport


class FakeFuture:
    def __init__(self, result: object) -> None:
        self._result = result

    def add_done_callback(self, callback: object) -> None:
        callback(self)

    def result(self) -> object:
        return self._result


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.available = True
        self.response = SimpleNamespace(ok=True, error_code="", message="", report_json="{}")

    def wait_for_service(self, timeout_sec: float) -> bool:
        assert timeout_sec > 0
        return self.available

    def call_async(self, request: object) -> FakeFuture:
        self.requests.append(request)
        return FakeFuture(self.response)


class FakePublisher:
    def __init__(self, node: "FakeNode", topic: str) -> None:
        self._node = node
        self._topic = topic
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)
        if self._topic == "/g1/arm/control/request_json":
            request = json.loads(message.data)
            callback = self._node.callbacks.get("/g1/arm/control/response_json")
            assert callback is not None
            callback(
                SimpleNamespace(
                    data=json.dumps(
                        {
                            "request_id": request["request_id"],
                            **self._node.arm_control_response,
                        }
                    )
                )
            )
            return
        if self._topic != "/g1/commissioning/request":
            return
        response = self._node.commissioning_response
        callback = self._node.callbacks.get("/g1/commissioning/response")
        if callback is None:
            return
        callback(
            SimpleNamespace(
                request_id=message.request_id,
                ok=response.ok,
                error_code=response.error_code,
                message=response.message,
                report_json=response.report_json,
            )
        )


class FakeNode:
    def __init__(self) -> None:
        self.publishers: list[FakePublisher] = []
        self.clients: list[FakeClient] = []
        self.callbacks: dict[str, object] = {}
        self.commissioning_response = SimpleNamespace(
            ok=True, error_code="", message="", report_json="{}"
        )
        self.arm_control_response = {
            "ok": True,
            "error_code": "",
            "message": "",
            "report": {"state": "ARMED"},
        }

    def create_publisher(self, message_type: type, topic: str, qos: object) -> FakePublisher:
        del message_type, qos
        publisher = FakePublisher(self, topic)
        self.publishers.append(publisher)
        return publisher

    def create_subscription(
        self, message_type: type, topic: str, callback: object, qos: object
    ) -> object:
        del message_type, qos
        self.callbacks[topic] = callback
        return object()

    def create_client(self, service_type: type, name: str) -> FakeClient:
        del service_type, name
        client = FakeClient()
        self.clients.append(client)
        return client

    def destroy_subscription(self, entity: object) -> None:
        del entity

    def destroy_client(self, entity: object) -> None:
        del entity

    def destroy_publisher(self, entity: object) -> None:
        del entity


class FakeRunner:
    def __init__(self) -> None:
        self.node = FakeNode()

    def start(self) -> FakeNode:
        return self.node

    def close(self) -> None:
        raise AssertionError("injected runners are not owned")


class Message:
    pass


class ArmControl:
    Request = Message


class CommissioningCommand:
    Request = Message


class CommissioningRequest(Message):
    def __init__(self) -> None:
        self.request_id = ""
        self.operation = ""
        self.request_json = ""


class CommissioningResponse(Message):
    pass


class Policy:
    KEEP_LAST = 1
    RELIABLE = 2
    BEST_EFFORT = 3
    VOLATILE = 4
    TRANSIENT_LOCAL = 5


def _types() -> dict[str, object]:
    return {
        "String": Message,
        "CommissioningState": Message,
        "CommissioningRequest": CommissioningRequest,
        "CommissioningResponse": CommissioningResponse,
        "CompressedDepth": Message,
        "CommissioningCommand": CommissioningCommand,
        "QoSProfile": lambda **kwargs: kwargs,
        "ReliabilityPolicy": Policy,
        "DurabilityPolicy": Policy,
        "HistoryPolicy": Policy,
    }


def test_target_and_service_calls_use_ros_entities() -> None:
    runner = FakeRunner()
    transport = RosTrackingTransport(runner=runner, types=_types())
    transport.start()

    assert transport.enable_arm("session", "calibration") == {"state": "ARMED"}
    transport.publish_target("session", 9, "calibration", [0.1] * 7, 12.4)
    transport.heartbeat_arm("session")
    transport.stop_arm("done")

    control = next(
        publisher
        for publisher in runner.node.publishers
        if publisher._topic == "/g1/arm/control/request_json"
    )
    assert [json.loads(message.data)["operation"] for message in control.messages] == [
        "enable",
        "heartbeat",
        "stop",
    ]
    target = runner.node.publishers[0].messages[0]
    target_payload = json.loads(target.data)
    assert target_payload["sequence"] == 9
    assert target_payload["pipeline_age_ms"] == 12
    assert target_payload["right_arm_q"] == [0.1] * 7
    transport.close()


@pytest.mark.skipif(
    importlib.util.find_spec("zstandard") is None,
    reason="zstandard is installed with the vision dependency group",
)
def test_depth_message_is_bounded_and_decoded() -> None:
    runner = FakeRunner()
    codec = DepthEnvelopeCodec(max_pixels=100, max_payload_size=4096)
    transport = RosTrackingTransport(
        runner=runner, types=_types(), codec=codec, monotonic=lambda: 4.5
    )
    transport.start()
    z16 = np.arange(12, dtype=np.uint16).reshape(3, 4)
    envelope = codec.encode(
        z16,
        sequence=7,
        width=4,
        height=3,
        depth_scale=0.001,
        sensor_timestamp_ms=12.0,
        timestamp_domain="sensor",
        calibration_id="calibration",
        registered_to_rgb=True,
    )
    decoded = codec.decode(envelope)
    message = SimpleNamespace(**as_message_fields(decoded.header, envelope))
    runner.node.callbacks["/g1/depth"](message)

    frame = transport.receive_depth(0.1)

    assert frame is not None
    assert frame.sequence == 7
    assert frame.receipt_time_s == 4.5
    assert frame.registered_to_rgb is True
    np.testing.assert_array_equal(frame.z16, z16)


def test_depth_only_observer_creates_no_arm_or_commissioning_entities() -> None:
    runner = FakeRunner()
    transport = RosTrackingTransport(
        runner=runner,
        types=_types(),
        observe_depth_only=True,
    )

    transport.start()

    assert set(runner.node.callbacks) == {"/g1/depth"}
    assert runner.node.publishers == []
    assert runner.node.clients == []
    assert transport.arm_state()["state"] == "unreachable"


def test_observe_only_subscribes_to_arm_state_without_command_publishers() -> None:
    runner = FakeRunner()
    transport = RosTrackingTransport(
        runner=runner,
        types=_types(),
        observe_only=True,
    )

    transport.start()

    assert set(runner.node.callbacks) == {"/g1/depth", "/g1/arm/state_json"}
    assert runner.node.publishers == []
    assert runner.node.clients == []


def test_observation_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RosTrackingTransport(observe_depth_only=True, observe_only=True)


def as_message_fields(header: object, envelope: bytes) -> dict[str, object]:
    header_size = int.from_bytes(envelope[:4], "big")
    payload = envelope[4 + header_size :]
    return {
        "version": header.version,
        "sequence": header.sequence,
        "width": header.width,
        "height": header.height,
        "depth_scale": header.depth_scale,
        "sensor_timestamp_ms": header.sensor_timestamp_ms,
        "timestamp_domain": header.timestamp_domain,
        "calibration_id": header.calibration_id,
        "registered_to_rgb": header.registered_to_rgb,
        "encoding": header.encoding,
        "uncompressed_size": header.uncompressed_size,
        "checksum_sha256": header.checksum_sha256,
        "payload": payload,
    }


def test_rejected_service_response_surfaces_robot_error() -> None:
    runner = FakeRunner()
    transport = RosTrackingTransport(runner=runner, types=_types())
    transport.start()
    runner.node.arm_control_response = {
        "ok": False,
        "error_code": "safety_gate",
        "message": "robot is not standing",
        "report": {},
    }

    try:
        transport.enable_arm("session", "calibration")
    except RosTrackingError as exc:
        assert exc.code == "safety_gate"
        assert "not standing" in str(exc)
    else:
        raise AssertionError("expected the rejected ROS service response to raise")


def test_commissioning_uses_topic_request_response() -> None:
    runner = FakeRunner()
    transport = RosTrackingTransport(runner=runner, types=_types())
    transport.start()
    runner.node.commissioning_response.report_json = json.dumps({"phase": "CREATED"})

    assert transport.commissioning("create", {"operator": "operator"}) == {"phase": "CREATED"}
    request = runner.node.publishers[1].messages[0]
    assert request.operation == "create"
    assert json.loads(request.request_json) == {"operator": "operator"}


def test_arm_state_topic_expires_using_local_monotonic_time() -> None:
    now = [10.0]
    runner = FakeRunner()
    transport = RosTrackingTransport(
        runner=runner,
        types=_types(),
        monotonic=lambda: now[0],
        state_topic_timeout_s=0.5,
    )
    transport.start()
    message = SimpleNamespace(
        data=json.dumps({"ok": True, "state": "ARMED", "session_id": "session"})
    )

    runner.node.callbacks["/g1/arm/state_json"](message)
    assert transport.arm_state()["state"] == "ARMED"

    now[0] += 0.501
    stale = transport.arm_state()
    assert stale["ok"] is False
    assert stale["state"] == "unreachable"
    assert stale["last_state"] == "ARMED"
    assert stale["topic_age_ms"] == 501.0
