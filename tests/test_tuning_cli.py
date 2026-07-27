import json

from object_tracking.tuning_cli import analyze_session


def test_analyze_session_reports_bottlenecks_and_safe_recommendations(tmp_path) -> None:
    session = tmp_path / "run-1"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "run-1",
                "mode": "dry-run",
                "expected_fps": 30,
                "configuration": {"imgsz": 960, "confidence": 0.35},
            }
        ),
        encoding="utf-8",
    )
    samples = []
    for index in range(10):
        samples.append(
            {
                "session_elapsed_s": index / 5,
                "fps": 20.0,
                "yolo_fps": 7.0,
                "detections": [],
                "tracks": [],
                "arm_tracking": {
                    "status": "rejected",
                    "reason": "rgb_depth_pair_stale",
                    "depth_age_ms": 175.0,
                    "pair_skew_ms": 90.0,
                },
            }
        )
    (session / "telemetry.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )

    report = analyze_session(session)

    assert report["sample_count"] == 10
    assert report["metrics"]["camera_fps"]["mean"] == 20.0
    assert report["metrics"]["rejection_reasons"] == {"rgb_depth_pair_stale": 10}
    assert report["configuration"]["imgsz"] == 960
    actions = " ".join(item["action"] for item in report["recommendations"])
    assert "do not raise" in actions.lower() or "do not loosen" in actions.lower()
