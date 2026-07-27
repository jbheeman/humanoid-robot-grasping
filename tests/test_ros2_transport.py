from __future__ import annotations

from types import SimpleNamespace

import pytest

from object_tracking.ros2_transport import (
    Ros2Bindings,
    Ros2NodeRunner,
    Ros2RequestResponseClient,
    Ros2RpcTimeout,
)


class Request:
    def __init__(self) -> None:
        self.header = SimpleNamespace(identity=SimpleNamespace(id=0, api_id=0))
        self.parameter = ""
        self.binary = []


class Response:
    def __init__(self, request_id: int, *, status: int = 0) -> None:
        self.header = SimpleNamespace(
            identity=SimpleNamespace(id=request_id, api_id=1001),
            status=SimpleNamespace(code=status),
        )
        self.data = '{"name":"ai"}'
        self.binary = [-1, 2]


class Publisher:
    def __init__(self, node: "Node", respond: bool) -> None:
        self.node = node
        self.respond = respond
        self.messages: list[Request] = []

    def publish(self, request: Request) -> None:
        self.messages.append(request)
        if self.respond:
            self.node.callback(Response(request.header.identity.id, status=7))


class Node:
    def __init__(self, *, respond: bool = True) -> None:
        self.publisher = Publisher(self, respond)
        self.callback = None
        self.destroyed: list[object] = []

    def create_publisher(self, message_type: type, topic: str, qos: int) -> Publisher:
        del message_type, topic, qos
        return self.publisher

    def create_subscription(self, message_type: type, topic: str, callback: object, qos: int):
        del message_type, topic, qos
        self.callback = callback
        return SimpleNamespace()

    def destroy_subscription(self, value: object) -> None:
        self.destroyed.append(value)

    def destroy_publisher(self, value: object) -> None:
        self.destroyed.append(value)


def test_request_response_client_correlates_identity_and_preserves_status() -> None:
    node = Node()
    ids = iter((123,))
    client = Ros2RequestResponseClient(
        node,
        request_type=Request,
        response_type=Response,
        request_topic="/request",
        response_topic="/response",
        monotonic_ns=lambda: next(ids),
    )

    result = client.call(api_id=1001, parameter='{"read":true}', binary=b"\xff", timeout_s=0.1)

    assert result.request_id == 123
    assert result.api_id == 1001
    assert result.status_code == 7
    assert result.data == '{"name":"ai"}'
    assert result.binary == b"\xff\x02"
    request = node.publisher.messages[0]
    assert request.header.identity.id == 123
    assert request.header.identity.api_id == 1001
    assert request.parameter == '{"read":true}'
    assert request.binary == [255]
    client.close()
    assert len(node.destroyed) == 2


def test_request_response_client_times_out_and_ignores_unknown_identity() -> None:
    node = Node(respond=False)
    client = Ros2RequestResponseClient(
        node,
        request_type=Request,
        response_type=Response,
        request_topic="/request",
        response_topic="/response",
        monotonic_ns=lambda: 44,
    )
    node.callback(Response(99))

    with pytest.raises(Ros2RpcTimeout, match="API 1001 timed out"):
        client.call(api_id=1001, timeout_s=0.001)


def test_node_runner_adapts_injected_node_without_owning_it() -> None:
    node = object()
    bindings = Ros2Bindings(
        rclpy=SimpleNamespace(),
        context_type=object,
        executor_type=object,
        low_cmd_type=object,
        low_state_type=object,
        request_type=Request,
        response_type=Response,
    )
    runner = Ros2NodeRunner("shared", bindings=bindings, node=node)

    assert runner.start() is node
    assert runner.node is node
    assert runner.bindings is bindings
    assert runner.owns_node is False
    runner.close()
