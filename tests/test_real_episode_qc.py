import json
from pathlib import Path

import pytest

from object_tracking.real_episode_qc import (
    audit_episode,
    grouped_cross_validation_folds,
    grouped_split,
)


def make_episode(
    root: Path,
    index: int,
    *,
    group: str,
    session: str,
    valid: bool = True,
) -> Path:
    directory = root / f"episode_{index:04d}"
    colors = directory / "colors"
    colors.mkdir(parents=True)
    frames = []
    for frame_index in range(20):
        image = colors / f"{frame_index:06d}_color_0.jpg"
        image.write_bytes(f"{index}-{frame_index}".encode())
        frames.append(
            {
                "idx": frame_index,
                "timing": {"monotonic_time_ns": frame_index * 33_333_333},
                "colors": {"color_0": f"colors/{image.name}"},
                "states": {"right_arm": {"qpos": [0.0] * 7}},
                "actions": {"right_arm": {"qpos": [0.1] * 7}},
                "annotations": {
                    "right_palm_contact": {"value": frame_index >= 15}
                },
            }
        )
    payload = {"info": {}, "data": frames}
    (directory / "data.json").write_text(json.dumps(payload))
    metadata = {
        "rabbit_path_angle_deg": 15 * (index % 5),
        "rabbit_speed_m_s": 0.15 + 0.01 * (index % 3),
        "right_arm_start": "tucked" if index % 2 else "neutral",
        "camera_view": "head",
        "collection_session_id": session,
        "collection_block_id": group,
        "moving_object": True,
        "contact_reviewed": valid,
        "visual_contact_frame": 15,
    }
    (directory / "episode_metadata.json").write_text(json.dumps(metadata))
    return directory


def test_valid_episode_enforces_timing_joint_and_visual_contact_contract(
    tmp_path: Path,
) -> None:
    audit = audit_episode(
        make_episode(tmp_path, 1, group="block-a", session="session-a")
    )
    assert audit.accepted
    assert audit.contact_frame == 15
    assert audit.condition_id is not None


def test_unreviewed_contact_is_rejected(tmp_path: Path) -> None:
    audit = audit_episode(
        make_episode(
            tmp_path, 1, group="block-a", session="session-a", valid=False
        )
    )
    assert not audit.accepted
    assert "contact_not_visually_reviewed" in audit.reasons


def test_group_split_never_leaks_collection_blocks(tmp_path: Path) -> None:
    audits = [
        audit_episode(
            make_episode(
                tmp_path,
                index,
                group=f"block-{index // 10}",
                session=f"session-{index // 10}",
            )
        )
        for index in range(30)
    ]
    assignments = grouped_split(audits)
    for group_index in range(3):
        splits = {
            assignments[f"episode_{index:04d}"]
            for index in range(group_index * 10, group_index * 10 + 10)
        }
        assert len(splits) == 1
    assert set(assignments.values()) == {"train", "val", "test"}


def test_pilot_cv_requires_independent_groups(tmp_path: Path) -> None:
    audits = [
        audit_episode(
            make_episode(tmp_path, index, group="one", session="one")
        )
        for index in range(20)
    ]
    with pytest.raises(ValueError, match="independent"):
        grouped_cross_validation_folds(audits)
