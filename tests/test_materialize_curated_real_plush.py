import json
from pathlib import Path

from scripts.training.materialize_curated_real_plush import materialize_episode


def test_materializer_trims_reset_tail_and_omits_unused_modalities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "episode_0042"
    colors = source / "colors"
    colors.mkdir(parents=True)
    frames = []
    for index in range(10):
        image = colors / f"{index:06d}_color_0.jpg"
        image.write_bytes(f"frame-{index}".encode())
        frames.append(
            {
                "idx": index,
                "colors": {"color_0": f"colors/{image.name}"},
                "depths": {"depth_0": f"depths/{index}.png"},
                "audios": {"audio_0": f"audios/{index}.wav"},
            }
        )
    (source / "data.json").write_text(json.dumps({"data": frames}))
    record = {
        "episode_id": 42,
        "episode": "episode_0042",
        "path": str(source),
        "status": "accepted",
        "reasons": [],
        "training_frame_range": {"start": 0, "end_exclusive": 7},
    }

    episode, files = materialize_episode(record, tmp_path / "curated")

    payload = json.loads(
        (tmp_path / "curated/episode_0042/data.json").read_text()
    )
    assert episode["frames"] == 7
    assert len(payload["data"]) == 7
    assert payload["data"][-1]["idx"] == 6
    assert payload["data"][-1]["depths"] == {}
    assert payload["data"][-1]["audios"] == {}
    assert payload["curation"]["human_reset_tail_excluded"]
    assert len(files) == 8


def test_materializer_can_extend_from_raw_contact_metrics(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "episode_0001"
    colors = source / "colors"
    colors.mkdir(parents=True)
    frames = []
    for index in range(20):
        image = colors / f"{index:06d}.jpg"
        image.write_bytes(str(index).encode())
        frames.append(
            {
                "idx": index,
                "colors": {"color_0": f"colors/{image.name}"},
                "depths": {},
                "audios": {},
            }
        )
    (source / "data.json").write_text(json.dumps({"data": frames}))
    record = {
        "episode_id": 1,
        "episode": "episode_0001",
        "path": str(source),
        "status": "accepted",
        "reasons": [],
        "metrics": {"first_contact_frame": 8},
        "training_frame_range": {"start": 0, "end_exclusive": 15},
    }
    episode, _files = materialize_episode(
        record,
        tmp_path / "curated",
        post_contact_frames=9,
    )
    assert episode["frames"] == 18
