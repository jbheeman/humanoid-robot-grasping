from object_tracking.vla_visual_safety import VisualSafetyGovernor


def track(**overrides):
    value = {
        "class_name": "rabbit_plush",
        "confidence": 0.9,
        "last_seen": 1.0,
        "age_frames": 5,
        "missed_updates": 0,
        "velocity_px_per_sec": [100.0, -20.0],
    }
    value.update(overrides)
    return value


def test_valid_fresh_target_does_not_veto() -> None:
    signal = VisualSafetyGovernor().signal(
        [track()], now_s=1.1, table_clearance_m=0.08
    )
    assert not signal.tracker_veto
    assert signal.tracker_confidence == 0.9


def test_missing_stale_or_implausible_target_vetoes() -> None:
    governor = VisualSafetyGovernor()
    assert governor.signal([], now_s=1.0, table_clearance_m=0.08).tracker_veto
    assert governor.signal(
        [track(last_seen=0.5)], now_s=1.0, table_clearance_m=0.08
    ).tracker_veto
    assert governor.signal(
        [track(velocity_px_per_sec=[2_000.0, 0.0])], now_s=1.1, table_clearance_m=0.08
    ).tracker_veto


def test_ambiguous_targets_veto_instead_of_choosing_one() -> None:
    signal = VisualSafetyGovernor().signal(
        [track(confidence=0.90), track(confidence=0.86)],
        now_s=1.1,
        table_clearance_m=0.08,
    )
    assert signal.tracker_veto


def test_contact_and_clearance_are_forwarded_to_scheduler() -> None:
    signal = VisualSafetyGovernor().signal(
        [track()], now_s=1.1, table_clearance_m=0.049, contact=True
    )
    assert signal.contact
    assert signal.table_clearance_m == 0.049
