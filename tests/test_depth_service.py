import time

import numpy as np

from scripts.robot.depth_service import (
    CapturedDepth,
    DepthService,
    RealSenseRgbRtpRelay,
)


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


def test_depth_service_closes_realsense_when_rgb_relay_start_fails() -> None:
    source = FakeDepthSource()
    source.calibration["color_profile"] = {"width": 4, "height": 3}

    class FailingRelay:
        def start(self, *, width: int, height: int) -> None:
            raise RuntimeError("relay failed")

        def close(self) -> None:
            pass

    service = DepthService(source, rgb_relay=FailingRelay())  # type: ignore[arg-type]
    try:
        service.start()
    except RuntimeError as exc:
        assert str(exc) == "relay failed"
    else:
        raise AssertionError("relay startup failure was not propagated")
    assert source.started is False


def test_rgb_relay_feeds_native_gstreamer_subprocess(monkeypatch) -> None:
    class Stdin:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, value: bytes) -> int:
            self.data.extend(value)
            return len(value)

        def close(self) -> None:
            pass

    class Process:
        def __init__(self) -> None:
            self.stdin = Stdin()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout: float):
            self.returncode = 0
            return 0

    process = Process()
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr("scripts.robot.depth_service.subprocess.Popen", fake_popen)
    relay = RealSenseRgbRtpRelay(host="192.168.0.66", port=5600, fps=60)
    relay.start(width=4, height=3)
    relay.write(np.zeros((3, 4, 3), dtype=np.uint8))
    relay.close()

    command = captured["command"]
    assert command[0:3] == ["gst-launch-1.0", "-q", "fdsrc"]
    assert "nvv4l2h264enc" in command
    assert "host=192.168.0.66" in command
    assert len(process.stdin.data) == 3 * 4 * 3
    assert relay.frames_sent == 1
