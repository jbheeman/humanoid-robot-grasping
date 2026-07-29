from fastapi.testclient import TestClient

from object_tracking import yolo_stream_server
from object_tracking.research_session import ResearchSession


class FakeRosTransport:
    def __init__(self) -> None:
        self.stops: list[str] = []
        self.returns: list[str] = []
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

    def return_arm(self, reason: str) -> dict:
        self.returns.append(reason)
        return {"state": "RETURNING", "return_reason": reason}

    def commissioning(self, command: str, payload: dict) -> dict:
        self.commissioning_calls.append((command, payload))
        return {"command": command, **payload}


class FakeArmRuntime:
    def __init__(self) -> None:
        self.profiles: list[str] = []
        self.reset_count = 0

    def set_follow_profile(self, profile: str) -> dict:
        self.profiles.append(profile)
        return {"ok": True, "changed": True, "follow_profile": profile}

    def reset_intercept(self) -> None:
        self.reset_count += 1


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


def test_dashboard_endpoint_is_compact_and_exposes_overlay_state() -> None:
    previous = yolo_stream_server.state.arm_tracking
    yolo_stream_server.state.arm_tracking = {
        "status": "target_sent",
        "arm_state": "ARMED",
        "pipeline_age_ms": 123.0,
        "visualization": {
            "camera": {"intrinsics": {"width": 960, "height": 540}},
            "support_plane": {"footprint": {"source": "test"}},
            "support_plane_status": {"available": True, "source": "test"},
            "predicted_trajectory_xyz_m": [[0.4, 0.0, 0.1]],
        },
        "large_unused_field": list(range(100)),
    }
    try:
        report = TestClient(yolo_stream_server.app).get("/api/v1/dashboard")
        assert report.status_code == 200
        payload = report.json()
        assert payload["tracking"]["arm_state"] == "ARMED"
        assert payload["visualization"]["support_plane_status"]["available"] is True
        assert "large_unused_field" not in payload["tracking"]
    finally:
        yolo_stream_server.state.arm_tracking = previous


def test_raw_snapshot_is_clean_latest_frame_with_freshness_headers() -> None:
    import numpy as np

    previous_frame = yolo_stream_server.state.raw_frame
    previous_received = yolo_stream_server.state.raw_frame_received_monotonic
    previous_count = yolo_stream_server.state.frame_count
    try:
        yolo_stream_server.state.raw_frame = np.full((8, 12, 3), 127, dtype=np.uint8)
        yolo_stream_server.state.raw_frame_received_monotonic = __import__("time").monotonic()
        yolo_stream_server.state.frame_count = 42
        response = TestClient(yolo_stream_server.app).get("/raw-snapshot.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-g1-frame-id"] == "42"
        assert float(response.headers["x-g1-frame-age-ms"]) >= 0.0
        assert response.headers["cache-control"] == "no-store"
    finally:
        yolo_stream_server.state.raw_frame = previous_frame
        yolo_stream_server.state.raw_frame_received_monotonic = previous_received
        yolo_stream_server.state.frame_count = previous_count


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


def test_tabletop_calibration_saves_ordered_tape_corners(tmp_path) -> None:
    previous = yolo_stream_server.TABLETOP_CALIBRATION_PATH
    yolo_stream_server.TABLETOP_CALIBRATION_PATH = tmp_path / "tabletop.json"
    try:
        client = TestClient(yolo_stream_server.app)
        assert client.get("/api/v1/tabletop-calibration").json() == {"configured": False}
        response = client.post(
            "/api/v1/tabletop-calibration",
            json={
                "width_m": 0.65,
                "depth_m": 0.40,
                "image_width": 960,
                "image_height": 540,
                "corners": [
                    {"name": "near_left", "x": 100, "y": 400},
                    {"name": "near_right", "x": 800, "y": 400},
                    {"name": "far_right", "x": 760, "y": 100},
                    {"name": "far_left", "x": 140, "y": 100},
                ],
            },
        )
        assert response.status_code == 200
        saved = response.json()["calibration"]
        assert saved["tabletop"] == {"width_m": 0.65, "depth_m": 0.4}
        assert [corner["name"] for corner in saved["corners_px"]] == [
            "near_left", "near_right", "far_right", "far_left"
        ]
        assert client.get("/api/v1/tabletop-calibration").json()["configured"] is True
    finally:
        yolo_stream_server.TABLETOP_CALIBRATION_PATH = previous


def test_tabletop_localization_maps_track_center_to_table_metres(tmp_path) -> None:
    previous_path = yolo_stream_server.TABLETOP_CALIBRATION_PATH
    previous_cache = yolo_stream_server._tabletop_cache
    previous_mtime = yolo_stream_server._tabletop_cache_mtime
    path = tmp_path / "tabletop.json"
    path.write_text(
        """{
  "camera_frame": {"width": 960, "height": 540},
  "tabletop": {"width_m": 0.65, "depth_m": 0.4},
  "corners_px": [
    {"name": "near_left", "x": 100, "y": 400},
    {"name": "near_right", "x": 800, "y": 400},
    {"name": "far_right", "x": 800, "y": 100},
    {"name": "far_left", "x": 100, "y": 100}
  ]
}""",
        encoding="utf-8",
    )
    yolo_stream_server.TABLETOP_CALIBRATION_PATH = path
    yolo_stream_server._tabletop_cache = None
    yolo_stream_server._tabletop_cache_mtime = None
    try:
        result = yolo_stream_server.tabletop_localization(
            [{"track_id": 7, "confidence": 0.9, "center_xy": [450, 250]}]
        )
        assert result["status"] == "localized"
        assert result["track_id"] == 7
        assert result["table_xy_m"] == [0.325, 0.2]
    finally:
        yolo_stream_server.TABLETOP_CALIBRATION_PATH = previous_path
        yolo_stream_server._tabletop_cache = previous_cache
        yolo_stream_server._tabletop_cache_mtime = previous_mtime


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
        returned = client.post(
            "/api/v1/arm/return-to-neutral",
            json={"reason": "ui_return"},
        )
        assert returned.status_code == 200
        assert returned.json()["state"] == "RETURNING"
        assert transport.returns == ["ui_return"]
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


def test_robot_selected_follow_profile_is_applied_before_enable() -> None:
    previous_transport = yolo_stream_server.tracking_transport
    previous_runtime = yolo_stream_server.arm_runtime
    transport = FakeRosTransport()
    runtime = FakeArmRuntime()
    yolo_stream_server.tracking_transport = transport
    yolo_stream_server.arm_runtime = runtime
    try:
        client = TestClient(yolo_stream_server.app)
        selected = client.post(
            "/api/v1/arm/follow-profile",
            json={"follow_profile": "aggressive"},
        )
        assert selected.status_code == 200
        assert selected.json()["follow_profile"] == "aggressive"

        enabled = client.post(
            "/api/v1/arm/enable",
            json={
                "session_id": "operator-1",
                "calibration_id": "cal-1",
                "follow_profile": "aggressive",
            },
        )
        assert enabled.status_code == 200
        assert runtime.profiles == ["aggressive", "aggressive"]
        assert runtime.reset_count == 1
        assert (
            client.post(
                "/api/v1/arm/follow-profile",
                json={"follow_profile": "reckless"},
            ).status_code
            == 422
        )
    finally:
        yolo_stream_server.tracking_transport = previous_transport
        yolo_stream_server.arm_runtime = previous_runtime


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
