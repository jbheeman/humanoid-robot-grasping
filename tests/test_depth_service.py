import time

import numpy as np

from scripts.robot.depth_service import CapturedDepth, DepthService


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


class FakeCodec:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, z16: np.ndarray, **metadata: object) -> bytes:
        self.calls.append({"z16": z16.copy(), **metadata})
        return b"bounded-compressed-depth"


def test_depth_service_prepares_latest_ros_payload_and_health() -> None:
    source = FakeDepthSource()
    codec = FakeCodec()
    service = DepthService(source, transmit_fps=15.0)
    service.codec = codec  # type: ignore[assignment]
    service.start()
    try:
        deadline = time.monotonic() + 1.0
        while service.latest_sequence < 0 and time.monotonic() < deadline:
            time.sleep(0.005)

        health = service.health()
        assert health["ok"] is True
        assert health["executable"] is True
        assert health["calibration_id"] == "test-calibration"
        assert service.latest_sequence == 0
        assert service.latest_envelope == b"bounded-compressed-depth"
        assert codec.calls[0]["registered_to_rgb"] is True
        assert codec.calls[0]["width"] == 4
        assert np.asarray(codec.calls[0]["z16"]).shape == (3, 4)
    finally:
        service.stop()

    assert source.started is False
