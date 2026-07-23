from object_tracking.real_visual_qc import (
    classify_episode,
    contact_runs,
    maximum_center_displacement_px,
    structural_reasons,
    training_frame_range,
    visual_reasons,
)


def metrics(**updates: object) -> dict[str, object]:
    value = {
        "frames": 150,
        "duration_s": 5.0,
        "sample_hz": 30.0,
        "missing_images": 0,
        "invalid_joint_frames": 0,
        "first_contact_frame": 120,
        "contact_transitions": 1,
    }
    value.update(updates)
    return value


def test_contact_runs_and_trim_exclude_human_reset_tail() -> None:
    assert contact_runs([False, True, True, False, True]) == ((1, 3), (4, 5))
    assert training_frame_range(200, 120) == (0, 127)


def test_missing_or_early_contact_is_automatic_reject() -> None:
    assert "no_contact_annotation" in structural_reasons(
        metrics(first_contact_frame=-1)
    )
    assert "contact_starts_too_early" in structural_reasons(
        metrics(first_contact_frame=3)
    )


def test_visual_qc_requires_visible_moving_plush_at_contact() -> None:
    stationary = [(10.0, 10.0), (12.0, 11.0), None]
    reasons = visual_reasons(centers=stationary, contact_visible=False)
    assert "plush_not_visible_at_contact" in reasons
    assert "plush_motion_too_small" in reasons

    moving = [(10.0, 10.0), (50.0, 12.0), (90.0, 15.0)]
    assert not visual_reasons(centers=moving, contact_visible=True)
    assert maximum_center_displacement_px(moving) > 70.0


def test_operator_rejection_always_wins() -> None:
    assert classify_episode(
        operator_rejected=True,
        structural=(),
        visual=(),
    ) == ("operator_reject", ("operator_rejected",))
