import json

from object_tracking.research_session import ResearchSession


def test_research_session_records_separate_jsonl_and_summary(tmp_path) -> None:
    session = ResearchSession(
        tmp_path / "runs" / "research" / "arm_tracking", metadata={"mode": "dry-run"}
    )
    session.record(
        {
            "fps": 30.0,
            "yolo_fps": 15.0,
            "detections": [{"class_name": "plushie"}],
            "tracks": [{"track_id": 1}],
            "arm_tracking": {
                "status": "rejected",
                "reason": "depth_uncertain",
                "depth_age_ms": 20.0,
                "pair_skew_ms": 8.0,
            },
        }
    )

    report = session.session_report()
    assert "runs/research/arm_tracking" in report["directory"]
    line = json.loads(session.telemetry_path.read_text(encoding="utf-8"))
    assert line["detections"][0]["class_name"] == "plushie"
    summary = session.summary_report()
    assert summary["sample_count"] == 1
    assert summary["rejection_reasons"] == {"depth_uncertain": 1}
