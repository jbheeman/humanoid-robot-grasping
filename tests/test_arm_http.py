from __future__ import annotations

from io import BytesIO
import json
from typing import Any

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmControlMode,
    RobotState,
)
from object_tracking.arm_tracking.arm_commissioning import (
    OPERATOR_ACK,
    CommissioningConfig,
    CommissioningController,
)
from object_tracking.arm_tracking.joints import joint_contract_id
from scripts.unitree_arm_bridge import make_handler


class Hardware:
    def __init__(self) -> None:
        self.state = RobotState(
            arm_q=(0.0,) * 14,
            received_at=10.0,
            standing=True,
            standing_since=0.0,
            motion_mode_verified=True,
            controller_ownership_verified=True,
            motor_status_verified=True,
            motor_state_healthy=True,
        )

    def start(self) -> None:
        pass

    def latest_state(self) -> RobotState:
        return self.state

    def publish(self, command: object) -> None:
        pass

    def close(self) -> None:
        pass


def dispatch(
    handler_type: type[Any],
    path: str,
    method: str,
    *,
    body: dict[str, object] | None = None,
    token: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Exercise the real request dispatch without opening a sandboxed socket."""
    raw = b"" if body is None else json.dumps(body).encode()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
    if token is not None:
        handler.headers["Authorization"] = f"Bearer {token}"
    handler.rfile = BytesIO(raw)
    sent: list[tuple[int, dict[str, object]]] = []
    handler.send_json = lambda status, report: sent.append((status, report))
    if method == "GET":
        handler.do_GET()
    else:
        handler.do_POST()
    assert len(sent) == 1
    return sent[0]


def test_rest_health_authentication_and_enable() -> None:
    hardware = Hardware()
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(allow_movement=True, calibration_id="cal-1"),
        monotonic=lambda: 10.0,
        wall_time=lambda: 1_800_000_000.0,
    )
    handler = make_handler(controller, "test-token")

    status, report = dispatch(handler, "/health", "GET")
    assert status == 200
    assert report["state"] == "DISARMED"
    assert "token" not in json.dumps(report).lower()

    status, report = dispatch(
        handler,
        "/arm/enable",
        "POST",
        body={"session_id": "session-a", "calibration_id": "cal-1"},
    )
    assert status == 401
    assert report["error"] == "unauthorized"

    status, report = dispatch(
        handler,
        "/arm/enable",
        "POST",
        body={"session_id": "session-a", "calibration_id": "cal-1"},
        token="test-token",
    )
    assert status == 200
    assert report["state"] == "ARMING"

    status, report = dispatch(handler, "/arm/stop", "POST", body={}, token="test-token")
    assert status == 200
    assert report["state"] == "HOLDING"


def test_commissioning_api_is_authenticated_and_mode_scoped(tmp_path) -> None:
    hardware = Hardware()
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            control_mode=ArmControlMode.COMMISSIONING,
            allow_movement=True,
            joint_contract_id=joint_contract_id(),
        ),
        monotonic=lambda: 10.0,
        wall_time=lambda: 1_800_000_000.0,
    )
    commissioning = CommissioningController(
        controller,
        CommissioningConfig(
            research_root=tmp_path / "runs",
            profile_path=tmp_path / "home.json",
        ),
        monotonic=lambda: 10.0,
    )
    handler = make_handler(controller, "test-token", commissioning)

    status, report = dispatch(handler, "/api/v1/commissioning/state", "GET")
    assert status == 401
    assert report["error"] == "unauthorized"

    status, report = dispatch(
        handler,
        "/api/v1/commissioning/sessions",
        "POST",
        body={
            "operator_ack": OPERATOR_ACK,
            "operator": "tester",
            "client_id": "pytest",
        },
        token="test-token",
    )
    assert status == 200
    session_id = report["session_id"]

    status, report = dispatch(
        handler,
        f"/api/v1/commissioning/sessions/{session_id}/enable",
        "POST",
        body={},
        token="test-token",
    )
    assert status == 200
    assert report["phase"] == "ARMING"

    status, report = dispatch(
        handler,
        "/arm/enable",
        "POST",
        body={"session_id": "tracking", "calibration_id": "cal-1"},
        token="test-token",
    )
    assert status == 403
    assert report["error"] == "mode_mismatch"
