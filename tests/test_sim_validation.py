from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_tracking.arm_tracking.sim_validation import (
    build_replay_episode,
    episode_windows,
    read_research_telemetry,
    summarize_episode,
)


def _row(
    elapsed: float,
    *,
    status: str,
    sequence: int = 0,
    reason: str | None = None,
    object_y: float = 0.1,
    target_y: float = 0.04,
) -> dict:
    arm = {
        "status": status,
        "reason": reason,
        "calibration_id": "cal-1",
        "target_sequence": sequence,
    }
    if status == "target_sent":
        arm.update(
            {
                "predicted_bounded_arm_command_rad": [0.1] * 7,
                "gravity_feedforward_tau_nm": [0.2] * 7,
                "object_xyz_m": [0.45, object_y, 0.1],
                "target_xyz_m": [0.39, target_y, 0.13],
                "pipeline_age_ms": 180.0,
                "visualization": {
                    "measured_pose_rad": [0.0] * 29,
                    "support_plane": {
                        "normal": [0.0, 0.0, 1.0],
                        "offset": 0.0,
                        "source": "automatic_rgbd_plane_calibrated_footprint_prior",
                    },
                },
            }
        )
    return {"session_elapsed_s": elapsed, "arm_tracking": arm}


def _telemetry(tmp_path: Path) -> Path:
    path = tmp_path / "telemetry.jsonl"
    rows = [
        _row(0.0, status="waiting_for_depth"),
        _row(1.0, status="target_sent", sequence=1),
        _row(1.2, status="target_sent", sequence=5),
        _row(1.4, status="rejected", sequence=5, reason="target_lost"),
        _row(2.4, status="rejected", sequence=5, reason="target_lost"),
        _row(8.0, status="target_sent", sequence=6),
        _row(8.2, status="target_sent", sequence=10),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_episode_detection_splits_long_command_gaps(tmp_path: Path) -> None:
    records = read_research_telemetry(_telemetry(tmp_path))
    assert episode_windows(records) == ((0.0, 3.2), (7.0, 8.2))


def test_replay_preserves_loss_tail_transform_and_right_side_contract(tmp_path: Path) -> None:
    episode = build_replay_episode(_telemetry(tmp_path), episode_index=0)
    report = summarize_episode(episode)
    assert [frame.status for frame in episode.frames] == [
        "waiting_for_depth",
        "target_sent",
        "target_sent",
        "rejected",
        "rejected",
    ]
    assert report["estimated_target_publish_hz"] == pytest.approx(20.0)
    assert report["right_side_margin_m"]["minimum"] == pytest.approx(0.06)
    assert report["right_side_margin_m"]["fraction_on_right"] == 1.0
    assert report["rejection_reasons"] == {"target_lost": 2}
    assert report["support_plane_source"].startswith("automatic_rgbd_")


def test_explicit_window_and_invalid_json_are_reported(tmp_path: Path) -> None:
    path = _telemetry(tmp_path)
    episode = build_replay_episode(path, window=(7.5, 8.3))
    assert len(episode.frames) == 2
    path.write_text("{broken\n")
    with pytest.raises(ValueError, match="line 1"):
        read_research_telemetry(path)
