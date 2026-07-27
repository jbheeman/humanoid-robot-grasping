from __future__ import annotations

import pytest

from object_tracking.arm_tracking.arm_bridge import ArmBridgeError
from scripts.robot.native_arm_worker import NativeArmService


class FakeController:
    def __init__(self) -> None:
        self.calls = []

    def enable(self, **values):
        self.calls.append(("enable", values))
        return {"state": "ARMING"}

    def heartbeat(self, **values):
        self.calls.append(("heartbeat", values))
        return {"state": "ARMED"}

    def set_target(self, **values):
        self.calls.append(("target", values))
        return {"last_sequence": values["sequence"]}

    def stop(self, reason):
        self.calls.append(("stop", reason))
        return {"state": "HOLDING"}

    def state_report(self):
        self.calls.append(("state", {}))
        return {"state": "DISARMED"}


def test_native_service_preserves_guarded_tracking_protocol() -> None:
    controller = FakeController()
    service = NativeArmService(controller)  # type: ignore[arg-type]

    assert service.dispatch({
        "operation": "enable",
        "payload": {"session_id": "s", "calibration_id": "c"},
    }) == {"state": "ARMING"}
    assert service.dispatch({
        "operation": "heartbeat", "payload": {"session_id": "s"}
    }) == {"state": "ARMED"}
    assert service.dispatch({
        "operation": "target",
        "payload": {
            "session_id": "s",
            "sequence": 7,
            "calibration_id": "c",
            "right_arm_q": [0.01] * 7,
            "pipeline_age_ms": 12,
        },
    }) == {"last_sequence": 7}
    assert service.dispatch({
        "operation": "stop", "payload": {"reason": "guard_complete"}
    }) == {"state": "HOLDING"}
    assert service.dispatch({"operation": "state", "payload": {}}) == {
        "state": "DISARMED"
    }

    target = controller.calls[2]
    assert target[0] == "target"
    assert target[1]["pipeline_age_ms"] == 12
    assert target[1]["right_arm_q"] == [0.01] * 7


def test_native_service_rejects_unknown_or_non_object_requests() -> None:
    service = NativeArmService(FakeController())  # type: ignore[arg-type]

    with pytest.raises(ArmBridgeError, match="JSON object"):
        service.dispatch([])
    with pytest.raises(ArmBridgeError, match="Unknown native arm operation"):
        service.dispatch({"operation": "dance", "payload": {}})
