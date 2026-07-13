from fastapi.testclient import TestClient

from object_tracking import yolo_stream_server
from object_tracking.research_session import ResearchSession


class FakeRosTransport:
    def __init__(self) -> None:
        self.stops: list[str] = []
        self.commissioning_calls: list[tuple[str, dict]] = []

    def arm_state(self) -> dict:
        return {"state": "DISARMED", "weight": 0.0}

    def enable_arm(self, session_id: str, calibration_id: str) -> dict:
        return {
            "state": "ARMED",
            "session_id": session_id,
            "calibration_id": calibration_id,
        }

    def stop_arm(self, reason: str) -> None:
        self.stops.append(reason)

    def commissioning(self, command: str, payload: dict) -> dict:
        self.commissioning_calls.append((command, payload))
        return {"command": command, **payload}


def test_research_endpoints_expose_session_summary_and_export(tmp_path) -> None:
    session = ResearchSession(tmp_path / "research", metadata={"mode": "dry-run"})
    session.record(
        {"fps": 20.0, "yolo_fps": 10.0, "detections": [], "tracks": [], "arm_tracking": {}}
    )
    previous = yolo_stream_server.research_session
    yolo_stream_server.research_session = session
    try:
        client = TestClient(yolo_stream_server.app)
        assert client.get("/research/session").json()["session_id"] == session.session_id
        assert client.get("/research/summary").json()["sample_count"] == 1
        assert len(client.get("/research/telemetry?limit=10").json()["samples"]) == 1
        export = client.get("/research/export.jsonl")
        assert export.status_code == 200
        assert export.headers["content-type"].startswith("application/x-ndjson")
    finally:
        yolo_stream_server.research_session = previous


def test_visualization_endpoint_is_read_only_and_reports_unavailable() -> None:
    previous = yolo_stream_server.state.arm_tracking
    yolo_stream_server.state.arm_tracking = {"visualization": {"schema_version": 1, "read_only": True, "available": False}}
    try:
        client = TestClient(yolo_stream_server.app)
        report = client.get("/visualization/state")
        assert report.status_code == 200
        assert report.json()["read_only"] is True
        assert client.post("/visualization/state", json={}).status_code == 405
    finally:
        yolo_stream_server.state.arm_tracking = previous


def test_single_server_serves_viewer_visual_assets_and_tokenless_commissioning() -> None:
    client = TestClient(yolo_stream_server.app)

    viewer = client.get("/")
    assert viewer.status_code == 200
    assert "G1 Grasping Research Console" in viewer.text
    assert client.get("/visual/manifest.json").status_code == 200

    commissioning = client.get("/commissioning/")
    assert commissioning.status_code == 200
    assert "G1 Right-Arm Commissioning" in commissioning.text
    assert "Bearer token" not in commissioning.text
    assert "Authorization" not in commissioning.text
    assert "HEARTBEAT_INTERVAL_MS = 200" in commissioning.text
    assert "startHeartbeat();" in commissioning.text
    assert "heartbeatInFlight" in commissioning.text
    assert 'window.addEventListener("pagehide", stopHeartbeat)' in commissioning.text
    assert "right_shoulder_yaw" in commissioning.text
    assert "right_elbow" not in commissioning.text
    assert "right_wrist_roll" not in commissioning.text


def test_ui_control_routes_forward_only_to_injected_ros_transport() -> None:
    previous = yolo_stream_server.tracking_transport
    transport = FakeRosTransport()
    yolo_stream_server.tracking_transport = transport
    try:
        client = TestClient(yolo_stream_server.app)
        assert client.get("/api/v1/arm/state").json()["state"] == "DISARMED"
        enabled = client.post(
            "/api/v1/arm/enable",
            json={"session_id": "operator-1", "calibration_id": "cal-1"},
        )
        assert enabled.status_code == 200
        assert enabled.json()["state"] == "ARMED"
        assert client.post("/api/v1/arm/stop", json={"reason": "ui_stop"}).status_code == 200
        assert transport.stops == ["ui_stop"]

        session = client.post(
            "/api/v1/commissioning/sessions",
            json={"operator": "test", "client_id": "browser"},
        )
        assert session.json()["command"] == "create_session"
        jog = client.post(
            "/api/v1/commissioning/sessions/session-1/jogs",
            json={"sequence": 1, "joint_name": "right_shoulder_pitch", "direction": 1},
        )
        assert jog.status_code == 200
        assert transport.commissioning_calls[-1] == (
            "jog",
            {
                "sequence": 1,
                "joint_name": "right_shoulder_pitch",
                "direction": 1,
                "session_id": "session-1",
            },
        )
    finally:
        yolo_stream_server.tracking_transport = previous


def test_ui_control_routes_fail_closed_without_ros_transport() -> None:
    previous = yolo_stream_server.tracking_transport
    yolo_stream_server.tracking_transport = None
    try:
        response = TestClient(yolo_stream_server.app).post(
            "/api/v1/arm/stop",
            json={"reason": "operator_stop"},
        )
        assert response.status_code == 503
    finally:
        yolo_stream_server.tracking_transport = previous
