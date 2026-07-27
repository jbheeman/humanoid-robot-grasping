from __future__ import annotations

import json
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
apply_splits = pytest.importorskip(
    "scripts.training.apply_canonical_real_splits"
).apply_splits


def test_apply_splits_moves_files_and_updates_metadata(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    original = root / "hdf5" / "real" / "train" / "episode_0001.hdf5"
    original.parent.mkdir(parents=True)
    with h5py.File(original, "w") as output:
        output.attrs["split"] = "train"
    manifest = {
        "episodes": [
            {
                "path": str(original),
                "source": "xr_teleoperate",
                "split": "train",
            }
        ]
    }
    (root / "CANONICAL_MANIFEST.json").write_text(json.dumps(manifest))
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "strategy": "capture_session_holdout",
                "episodes": {"episode_0001": "test"},
            }
        )
    )

    assert apply_splits(root, splits) == {"test": 1, "train": 0, "validation": 0}
    destination = root / "hdf5" / "real" / "test" / original.name
    assert destination.is_file()
    with h5py.File(destination, "r") as output:
        assert output.attrs["split"] == "test"
    updated = json.loads((root / "CANONICAL_MANIFEST.json").read_text())
    assert updated["episodes"][0]["path"] == str(destination)
    assert updated["episodes"][0]["split"] == "test"
    assert updated["real_split_strategy"] == "capture_session_holdout"
