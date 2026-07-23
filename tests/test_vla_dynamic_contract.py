import numpy as np

from scripts.training.audit_vla_dynamic_contract import (
    lag_errors,
    padded_action_chunks,
    pose17_to_pose23,
)


def test_lag_audit_recovers_causal_two_frame_delay() -> None:
    state = np.stack(
        [np.array([index, index * 0.5, 0.2]) for index in range(12)]
    )
    action = np.concatenate((state[2:], np.repeat(state[-1:], 2, axis=0)))
    errors = lag_errors(action, state, range(-2, 5))
    best = min(errors, key=lambda lag: np.median(errors[lag]))
    assert best == 2
    assert np.median(errors[best]) == 0


def test_action_chunks_repeat_terminal_absolute_target() -> None:
    actions = np.arange(12, dtype=np.float32).reshape(4, 3)
    chunks = padded_action_chunks(actions, 3)
    assert chunks.shape == (4, 3, 3)
    assert np.array_equal(chunks[-1], np.repeat(actions[-1:], 3, axis=0))
    assert np.array_equal(chunks[1, 0], actions[1])
    assert np.array_equal(chunks[1, 2], actions[3])


def test_pose17_conversion_preserves_right_xyz_and_shape() -> None:
    pose17 = np.zeros((2, 17), dtype=np.float32)
    pose17[:, 6:9] = (0.3, -0.1, 0.2)
    pose23 = pose17_to_pose23(pose17)
    assert pose23.shape == (2, 23)
    assert np.allclose(pose23[:, 9:12], pose17[:, 6:9])
