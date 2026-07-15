import numpy as np
import unittest

from g1_bunny_vla.contract import DatasetContract, FrameSample, ee17_to_unifolm23
from g1_bunny_vla.curriculum import sample_trajectory


class ContractTests(unittest.TestCase):
    def test_identity_rpy_converts_to_rotation_columns(self):
        values = np.zeros((2, 17), dtype=np.float32)
        converted = ee17_to_unifolm23(values)
        self.assertEqual(converted.shape, (2, 23))
        expected_rotation = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(converted[0, 3:9], expected_rotation)
        np.testing.assert_allclose(converted[0, 12:18], expected_rotation)

    def test_curriculum_is_seeded_and_speed_bounded(self):
        first = sample_trajectory("varied_motion", 17)
        second = sample_trajectory("varied_motion", 17)
        self.assertEqual(first, second)
        _, velocity = first.state_at(0.0)
        self.assertLess(np.linalg.norm(velocity), 0.3)

    def test_hand_push_stages_stay_within_measured_speed_bounds(self):
        for stage in ("slow_linear", "varied_motion"):
            for seed in range(20):
                velocity = sample_trajectory(stage, seed).velocity_xyz
                self.assertLessEqual(abs(velocity[1]), 0.15)
                self.assertGreaterEqual(abs(velocity[1]), 0.03)

    def test_frame_rejects_missing_camera(self):
        contract = DatasetContract()
        sample = FrameSample(
            timestamp=0.0,
            images={},
            qpos=np.zeros(contract.qpos_dim, np.float32),
            qvel=np.zeros(contract.qpos_dim, np.float32),
            action=np.zeros(contract.qpos_dim, np.float32),
            ee_qpos=np.zeros(contract.ee_dim, np.float32),
            ee_action=np.zeros(contract.ee_dim, np.float32),
            plush_position=np.zeros(3, np.float32),
            plush_velocity=np.zeros(3, np.float32),
        )
        with self.assertRaisesRegex(ValueError, "missing cameras"):
            sample.validate()


if __name__ == "__main__":
    unittest.main()
