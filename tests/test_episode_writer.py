import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from g1_bunny_vla.contract import CAMERA_NAMES, DatasetContract, FrameSample
from g1_bunny_vla.episode_writer import EpisodeWriter
from g1_bunny_vla.validate_dataset import validate_episode


class EpisodeWriterTests(unittest.TestCase):
    def test_round_trip_passes_validation(self):
        contract = DatasetContract()
        image = np.zeros((contract.image_height, contract.image_width, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000000.hdf5"
            writer = EpisodeWriter(path, "Stop the moving bunny.", metadata={"trajectory_seed": 3})
            for index in range(2):
                writer.append(
                    FrameSample(
                        timestamp=index / contract.fps,
                        images={name: image for name in CAMERA_NAMES},
                        qpos=np.zeros(contract.qpos_dim, np.float32),
                        qvel=np.zeros(contract.qpos_dim, np.float32),
                        action=np.zeros(contract.qpos_dim, np.float32),
                        ee_qpos=np.zeros(contract.ee_dim, np.float32),
                        ee_action=np.zeros(contract.ee_dim, np.float32),
                        plush_position=np.zeros(3, np.float32),
                        plush_velocity=np.zeros(3, np.float32),
                    ),
                    reasoning="track and approach",
                )
            writer.close(success=True)
            self.assertEqual(validate_episode(path), [])
            with h5py.File(path, "r") as root:
                self.assertEqual(root["action"].shape, (2, 19))
                self.assertEqual(root["observations/ee_qpos"].shape, (2, 17))
                self.assertTrue(root.attrs["success"])


if __name__ == "__main__":
    unittest.main()

