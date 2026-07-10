from fastapi.testclient import TestClient

from object_tracking import yolo_stream_server
from object_tracking.research_session import ResearchSession


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
