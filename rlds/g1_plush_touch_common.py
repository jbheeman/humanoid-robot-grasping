"""Shared TFDS/RLDS builder for canonical G1 plush-touch datasets."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds

from g1_bunny_vla.canonical import INSTRUCTION, pose17_to_pose23
from object_tracking.vla_target_alignment import (
    FUTURE_STATE_TARGET_V1,
    align_achieved_future_state,
)


class G1PlushTouchBuilderBase(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    SOURCE: str

    def _target_lookahead_frames(self) -> int:
        raw = os.environ.get("G1_PLUSH_FUTURE_STATE_LOOKAHEAD", "0")
        try:
            lookahead = int(raw)
        except ValueError as exc:
            raise ValueError(
                "G1_PLUSH_FUTURE_STATE_LOOKAHEAD must be an integer"
            ) from exc
        if lookahead < 0:
            raise ValueError("G1_PLUSH_FUTURE_STATE_LOOKAHEAD cannot be negative")
        return lookahead

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image_left_top": tfds.features.Image(
                                    shape=(480, 640, 3),
                                    dtype=np.uint8,
                                    encoding_format="jpeg",
                                ),
                                "state": tfds.features.Tensor(
                                    shape=(19,), dtype=np.float32
                                ),
                                "ee_state": tfds.features.Tensor(
                                    shape=(17,), dtype=np.float32
                                ),
                                "ee_state_6d": tfds.features.Tensor(
                                    shape=(23,), dtype=np.float32
                                ),
                            },
                            "action": tfds.features.Tensor(
                                shape=(19,), dtype=np.float32
                            ),
                            "ee_action": tfds.features.Tensor(
                                shape=(17,), dtype=np.float32
                            ),
                            "ee_action_6d": tfds.features.Tensor(
                                shape=(23,), dtype=np.float32
                            ),
                            "discount": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": {
                        "file_path": tfds.features.Text(),
                        "source": tfds.features.Text(),
                        "action_target_version": tfds.features.Text(),
                        "target_lookahead_frames": np.int64,
                    },
                }
            )
        )

    def _split_generators(self, dl_manager):  # noqa: ARG002
        canonical = os.environ.get("G1_PLUSH_CANONICAL_ROOT")
        if not canonical:
            raise ValueError("Set G1_PLUSH_CANONICAL_ROOT to the canonical dataset")
        root = Path(canonical).resolve() / "hdf5" / self.SOURCE
        split_dirs = {"train": "train", "val": "validation", "test": "test"}
        result = {}
        for tfds_name, directory in split_dirs.items():
            paths = sorted((root / directory).glob("*.hdf5"))
            if not paths:
                raise ValueError(
                    f"No {self.SOURCE}/{directory} HDF5 episodes under {root}"
                )
            result[tfds_name] = self._generate_examples(paths)
        return result

    def _generate_examples(self, paths: list[Path]):
        lookahead = self._target_lookahead_frames()
        for path in paths:
            with h5py.File(path, "r") as root:
                states = np.asarray(
                    root["observations/qpos"][:], dtype=np.float32
                )
                ee_states = np.asarray(
                    root["observations/ee_qpos"][:], dtype=np.float32
                )
                images = np.asarray(
                    root["observations/images/cam_left_high"][:], dtype=np.uint8
                )
                raw_language = root["language_raw"][()]
                language = (
                    raw_language.decode("utf-8")
                    if isinstance(raw_language, bytes)
                    else str(raw_language)
                )
                if lookahead:
                    joint_aligned = align_achieved_future_state(states, lookahead)
                    ee_aligned = align_achieved_future_state(
                        ee_states, lookahead
                    )
                    states = joint_aligned.observations
                    actions = joint_aligned.targets
                    ee_states = ee_aligned.observations
                    ee_actions = ee_aligned.targets
                    images = images[:-lookahead]
                    target_version = FUTURE_STATE_TARGET_V1
                else:
                    actions = np.asarray(root["action"][:], dtype=np.float32)
                    ee_actions = np.asarray(
                        root["ee_action"][:], dtype=np.float32
                    )
                    target_version = "recorded_command_v1"
            if language != INSTRUCTION:
                raise ValueError(f"{path}: unexpected language {language!r}")
            if not (
                len(states)
                == len(ee_states)
                == len(actions)
                == len(ee_actions)
                == len(images)
            ):
                raise ValueError(f"{path}: inconsistent aligned sequence lengths")
            ee_states_6d = pose17_to_pose23(ee_states)
            ee_actions_6d = pose17_to_pose23(ee_actions)
            count = len(actions)
            steps = []
            for index in range(count):
                steps.append(
                    {
                        "observation": {
                            "image_left_top": images[index],
                            "state": states[index],
                            "ee_state": ee_states[index],
                            "ee_state_6d": ee_states_6d[index],
                        },
                        "action": actions[index],
                        "ee_action": ee_actions[index],
                        "ee_action_6d": ee_actions_6d[index],
                        "discount": np.float32(1.0),
                        "is_first": index == 0,
                        "is_last": index == count - 1,
                        "is_terminal": index == count - 1,
                        "language_instruction": language,
                    }
                )
            yield f"{self.SOURCE}:{path.stem}", {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(path),
                    "source": self.SOURCE,
                    "action_target_version": target_version,
                    "target_lookahead_frames": np.int64(lookahead),
                },
            }
