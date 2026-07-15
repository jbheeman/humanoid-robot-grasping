"""TFDS/RLDS builder for G1 bunny-stop HDF5 episodes.

Place beside the official UniFoLM `conversion_utils.py`, then run `tfds build`.
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import tensorflow_datasets as tfds


def _rpy_to_6d(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(np.asarray(rpy, dtype=np.float32), -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    first = np.stack((cy * cp, sy * cp, -sp), axis=-1)
    second = np.stack((cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr), axis=-1)
    return np.concatenate((first, second), axis=-1).astype(np.float32)


def _ee17_to_23(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    left = np.concatenate((values[..., :3], _rpy_to_6d(values[..., 3:6])), axis=-1)
    right = np.concatenate((values[..., 6:9], _rpy_to_6d(values[..., 9:12])), axis=-1)
    return np.concatenate((left, right, values[..., 12:17]), axis=-1).astype(np.float32)


class G1BunnyStop(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        image = lambda doc: tfds.features.Image(  # noqa: E731
            shape=(480, 640, 3), dtype=np.uint8, encoding_format="jpeg", doc=doc
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image_left_top": image("Left head RGB observation."),
                                "image_right_top": image("Right head RGB observation."),
                                "image_left_wrist": image("Left wrist RGB observation."),
                                "image_right_wrist": image("Right wrist RGB observation."),
                                "state": tfds.features.Tensor(shape=(19,), dtype=np.float32),
                                "ee_state": tfds.features.Tensor(shape=(17,), dtype=np.float32),
                                "ee_state_6d": tfds.features.Tensor(shape=(23,), dtype=np.float32),
                            },
                            "action": tfds.features.Tensor(shape=(19,), dtype=np.float32),
                            "ee_action": tfds.features.Tensor(shape=(17,), dtype=np.float32),
                            "ee_action_6d": tfds.features.Tensor(shape=(23,), dtype=np.float32),
                            "discount": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": {"file_path": tfds.features.Text()},
                }
            )
        )

    def _split_generators(self, dl_manager):  # noqa: ARG002
        pattern = os.environ.get("G1_BUNNY_HDF5_GLOB")
        if not pattern:
            raise ValueError("Set G1_BUNNY_HDF5_GLOB to the episode_*.hdf5 files")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise ValueError(f"No HDF5 episodes matched {pattern!r}")
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths):
        cameras = {
            "image_left_top": "cam_left_high",
            "image_right_top": "cam_right_high",
            "image_left_wrist": "cam_left_wrist",
            "image_right_wrist": "cam_right_wrist",
        }
        for path in paths:
            with h5py.File(path, "r") as root:
                states = root["observations/qpos"][:]
                ee_states = root["observations/ee_qpos"][:]
                actions = root["action"][:]
                ee_actions = root["ee_action"][:]
                frames = {key: root[f"observations/images/{value}"][:] for key, value in cameras.items()}
                raw_language = root["language_raw"][()]
                language = raw_language.decode("utf-8") if isinstance(raw_language, bytes) else str(raw_language)
                count = actions.shape[0]
                steps = []
                ee_states_6d = _ee17_to_23(ee_states)
                ee_actions_6d = _ee17_to_23(ee_actions)
                for i in range(count):
                    steps.append(
                        {
                            "observation": {
                                **{key: values[i] for key, values in frames.items()},
                                "state": states[i],
                                "ee_state": ee_states[i],
                                "ee_state_6d": ee_states_6d[i],
                            },
                            "action": actions[i],
                            "ee_action": ee_actions[i],
                            "ee_action_6d": ee_actions_6d[i],
                            "discount": np.float32(1.0),
                            "is_first": i == 0,
                            "is_last": i == count - 1,
                            "is_terminal": i == count - 1,
                            "language_instruction": language,
                        }
                    )
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}
