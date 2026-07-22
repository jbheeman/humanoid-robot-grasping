"""Streaming HDF5 writer compatible with the official UniFoLM converter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contract import CAMERA_NAMES, DatasetContract, FrameSample


class EpisodeWriter:
    def __init__(
        self,
        path: str | Path,
        language_instruction: str,
        *,
        contract: DatasetContract = DatasetContract(),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("EpisodeWriter requires h5py; install the project dependencies") from exc

        self._h5py = h5py
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self._root = h5py.File(self.path, "w", libver="latest")
        self._length = 0
        self._last_timestamp: float | None = None
        self._root.attrs.update({"sim": True, "fps": contract.fps, **(metadata or {})})
        obs = self._root.create_group("observations")
        images = obs.create_group("images")
        # This task has one physical head camera.  Preserve Unitree's frozen
        # four-key HDF5 interface with hard links rather than materializing the
        # same pixels four times.  Existing converters still see every camera
        # path while HDF5 stores and appends only one dataset.
        self._primary_camera = CAMERA_NAMES[0]
        primary = images.create_dataset(
            self._primary_camera,
            shape=(0, contract.image_height, contract.image_width, 3),
            maxshape=(None, contract.image_height, contract.image_width, 3),
            chunks=(1, contract.image_height, contract.image_width, 3),
            dtype="uint8",
            compression="gzip",
            compression_opts=1,
        )
        for name in CAMERA_NAMES[1:]:
            images[name] = primary
        self._root.attrs.update(
            {
                "physical_camera_count": 1,
                "primary_camera": self._primary_camera,
                "camera_aliases_are_hard_links": True,
            }
        )
        self._vector_dataset(obs, "qpos", contract.qpos_dim)
        self._vector_dataset(obs, "qvel", contract.qpos_dim)
        self._vector_dataset(obs, "ee_qpos", contract.ee_dim)
        self._vector_dataset(self._root, "action", contract.qpos_dim)
        self._vector_dataset(self._root, "ee_action", contract.ee_dim)
        signals = self._root.create_group("sim_signals")
        self._vector_dataset(signals, "plush_position", 3)
        self._vector_dataset(signals, "plush_velocity", 3)
        self._vector_dataset(signals, "contact_force", 6)
        for name in ("tracking_valid", "grasped", "safety_event"):
            signals.create_dataset(name, shape=(0,), maxshape=(None,), chunks=True, dtype="bool")
        self._root.create_dataset("timestamp", shape=(0,), maxshape=(None,), chunks=True, dtype="float64")
        string_type = h5py.string_dtype("utf-8")
        self._root.create_dataset("language_raw", data=language_instruction, dtype=string_type)
        self._reasonings = self._root.create_dataset(
            "substep_reasonings", shape=(0,), maxshape=(None,), chunks=True, dtype=string_type
        )

    @staticmethod
    def _vector_dataset(group: Any, name: str, width: int) -> None:
        group.create_dataset(
            name,
            shape=(0, width),
            maxshape=(None, width),
            chunks=(256, width),
            dtype="float32",
            compression="gzip",
            compression_opts=1,
        )

    def append(self, sample: FrameSample, reasoning: str = "") -> None:
        sample.validate(self.contract)
        if self._last_timestamp is not None and sample.timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        i = self._length
        self._length += 1
        self._last_timestamp = sample.timestamp

        self._append(
            self._root[f"observations/images/{self._primary_camera}"],
            sample.images[self._primary_camera],
        )
        for name in ("qpos", "qvel", "ee_qpos"):
            self._append(self._root[f"observations/{name}"], getattr(sample, name))
        self._append(self._root["action"], sample.action)
        self._append(self._root["ee_action"], sample.ee_action)
        for name in ("plush_position", "plush_velocity", "contact_force"):
            self._append(self._root[f"sim_signals/{name}"], getattr(sample, name))
        for name in ("tracking_valid", "grasped", "safety_event"):
            self._append(self._root[f"sim_signals/{name}"], getattr(sample, name))
        self._append(self._root["timestamp"], sample.timestamp)
        self._append(self._reasonings, reasoning)
        if i % self.contract.fps == 0:
            self._root.flush()

    @staticmethod
    def _append(dataset: Any, value: Any) -> None:
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def close(self, *, success: bool | None = None) -> None:
        if self._root:
            self._root.attrs["total_frames"] = self._length
            if success is not None:
                self._root.attrs["success"] = bool(success)
            self._root.flush()
            self._root.close()

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(success=False if exc_type else None)
