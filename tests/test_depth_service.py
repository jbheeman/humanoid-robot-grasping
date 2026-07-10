import time

import numpy as np
from fastapi.testclient import TestClient

from scripts.unitree_depth_service import CapturedDepth, DepthService, create_app


class FakeDepthSource:
    def __init__(self) -> None:
        self.started = False
        self.sent = False
        self.calibration = {
            "calibration_id": "test-calibration",
            "registration_validated": True,
            "aligned_to_rgb": True,
            "depth_profile": {
                "width": 4,
                "height": 3,
                "depth_scale": 0.001,
            },
        }

    def start(self) -> None:
        self.started = True

    def read(self, timeout_s: float) -> CapturedDepth | None:
        if self.sent:
            time.sleep(min(timeout_s, 0.005))
            return None
        self.sent = True
        return CapturedDepth(
            np.full((3, 4), 1000, dtype=np.uint16),
            sensor_timestamp_ms=123.0,
            timestamp_domain="fake",
        )

    def close(self) -> None:
        self.started = False


def test_depth_service_health_calibration_and_websocket() -> None:
    source = FakeDepthSource()
    service = DepthService(source, transmit_fps=15.0)
    service.start()
    try:
        deadline = time.monotonic() + 1.0
        while service.latest_sequence < 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        client = TestClient(create_app(service))
        health = client.get("/health").json()
        assert health["ok"] is True
        assert health["executable"] is True
        assert client.get("/depth/calibration").json()["calibration_id"] == "test-calibration"
        with client.websocket_connect("/depth/stream") as websocket:
            decoded = service.codec.decode(websocket.receive_bytes())
        assert decoded.header.sequence == 0
        assert decoded.as_numpy().shape == (3, 4)
    finally:
        service.stop()
