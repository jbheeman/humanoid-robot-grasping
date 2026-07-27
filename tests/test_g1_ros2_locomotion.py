from __future__ import annotations

import json

import pytest

from object_tracking.g1_loco_cli import _require_execute, build_parser
from object_tracking.ros2_transport import Ros2RpcResult
from object_tracking.unitree_g1 import (
    G1Ros2LocoClient,
    LOCO_API_GET_FSM_ID,
    LOCO_API_SET_FSM_ID,
    LOCO_API_SET_VELOCITY,
    MOTION_API_CHECK_MODE,
    MOTION_API_SELECT_MODE,
    MotionSwitcherRos2Client,
    UnitreeG1Error,
)


class FakeRpcClient:
    def __init__(self, statuses: dict[int, int] | None = None) -> None:
        self.statuses = statuses or {}
        self.calls: list[dict[str, object]] = []

    def call(
        self,
        *,
        api_id: int,
        parameter: str = "",
        binary: bytes = b"",
        timeout_s: float = 1.0,
    ) -> Ros2RpcResult:
        self.calls.append(
            {
                "api_id": api_id,
                "parameter": parameter,
                "binary": binary,
                "timeout_s": timeout_s,
            }
        )
        return Ros2RpcResult(
            request_id=len(self.calls),
            api_id=api_id,
            status_code=self.statuses.get(api_id, 0),
            data='{"data":500}' if api_id == LOCO_API_GET_FSM_ID else "",
            binary=b"",
        )

    def close(self) -> None:
        return None


class FakeRunner:
    def __init__(self) -> None:
        self.bindings = type(
            "Bindings", (), {"request_type": object, "response_type": object}
        )()
        self.node = object()

    def start(self) -> object:
        return self.node


def test_ros_clients_use_unitree_api_topic_pairs() -> None:
    created: list[dict[str, object]] = []

    def factory(_node: object, **kwargs: object) -> FakeRpcClient:
        created.append(kwargs)
        return FakeRpcClient()

    G1Ros2LocoClient(runner=FakeRunner(), rpc_factory=factory).start()

    assert [(item["request_topic"], item["response_topic"]) for item in created] == [
        ("/api/sport/request", "/api/sport/response"),
        ("/api/ai_sport/request", "/api/ai_sport/response"),
    ]


def test_auto_probe_is_read_only_and_falls_back_to_ai_sport() -> None:
    sport = FakeRpcClient({LOCO_API_GET_FSM_ID: 3103})
    ai_sport = FakeRpcClient()
    client = G1Ros2LocoClient(
        clients={"sport": sport, "ai_sport": ai_sport},
        loco_service_name="auto",
    )

    result = client.probe_loco()

    assert result.returncode == 0
    assert json.loads(result.stdout)["selected_service"] == "ai_sport"
    assert [call["api_id"] for call in sport.calls] == [LOCO_API_GET_FSM_ID]
    assert [call["api_id"] for call in ai_sport.calls] == [LOCO_API_GET_FSM_ID]


def test_mutating_command_first_checks_service_and_uses_expected_payload() -> None:
    sport = FakeRpcClient()
    client = G1Ros2LocoClient(clients={"sport": sport}, loco_service_name="sport")

    result = client.damp()

    assert result.returncode == 0
    assert [call["api_id"] for call in sport.calls] == [
        LOCO_API_GET_FSM_ID,
        LOCO_API_SET_FSM_ID,
    ]
    assert json.loads(str(sport.calls[-1]["parameter"])) == {"data": 1}


def test_move_is_bounded_and_always_finishes_with_stop() -> None:
    sport = FakeRpcClient()
    client = G1Ros2LocoClient(
        clients={"sport": sport},
        loco_service_name="sport",
        sleep=lambda _seconds: None,
    )

    client.move(0.1, -0.05, 0.2, duration=0.4)

    assert [call["api_id"] for call in sport.calls] == [
        LOCO_API_GET_FSM_ID,
        LOCO_API_SET_VELOCITY,
        LOCO_API_SET_VELOCITY,
    ]
    move_payload = json.loads(str(sport.calls[-2]["parameter"]))
    stop_payload = json.loads(str(sport.calls[-1]["parameter"]))
    assert move_payload == {"duration": 0.4, "velocity": [0.1, -0.05, 0.2]}
    assert stop_payload["velocity"] == [0.0, 0.0, 0.0]

    with pytest.raises(UnitreeG1Error, match="vx must be within"):
        client.move(0.6, 0.0, 0.0, duration=0.2)


def test_motion_switcher_check_and_select_use_ros_api_contract() -> None:
    rpc = FakeRpcClient()
    client = MotionSwitcherRos2Client(rpc_client=rpc)

    client.check_mode()
    client.select_mode("ai")

    assert [call["api_id"] for call in rpc.calls] == [
        MOTION_API_CHECK_MODE,
        MOTION_API_SELECT_MODE,
    ]
    assert json.loads(str(rpc.calls[-1]["parameter"])) == {"name": "ai"}


def test_cli_requires_explicit_execute_only_for_motion() -> None:
    parser = build_parser()
    _require_execute(parser.parse_args(["probe"]))
    _require_execute(parser.parse_args(["stop_move"]))

    with pytest.raises(UnitreeG1Error, match="without --execute"):
        _require_execute(parser.parse_args(["move", "--velocity", "0.1 0 0 0.2"]))

    _require_execute(
        parser.parse_args(["move", "--velocity", "0.1 0 0 0.2", "--execute"])
    )
