import numpy as np

from object_tracking.vla_chunk_scheduler import (
    ScheduleDecision,
    SchedulerConfig,
    SafetySignal,
    VLAChunkScheduler,
)


def scheduler() -> VLAChunkScheduler:
    return VLAChunkScheduler(
        2,
        SchedulerConfig(
            max_observation_age_s=0.5,
            max_inference_latency_s=0.45,
            blend_steps=2,
            maximum_step=0.1,
        ),
    )


def test_late_chunk_is_rejected_and_robot_holds() -> None:
    subject = scheduler()
    result = subject.submit([[0.1, 0.1]], observation_time_s=0.0,
                            inference_completed_time_s=0.46, now_s=0.46)
    assert result.decision is ScheduleDecision.REJECT
    assert result.reason == "late_inference"
    assert subject.tick(now_s=0.47, safety=SafetySignal()).decision is ScheduleDecision.HOLD


def test_newer_chunk_replaces_tail_and_blends() -> None:
    subject = scheduler()
    subject.submit([[0.05, 0.0], [0.1, 0.0], [0.2, 0.0]],
                   observation_time_s=1.0, inference_completed_time_s=1.1, now_s=1.1)
    first = subject.tick(now_s=1.11, safety=SafetySignal())
    assert np.allclose(first.action, [0.05, 0.0])
    subject.submit([[0.20, 0.0], [0.30, 0.0]],
                   observation_time_s=1.2, inference_completed_time_s=1.3, now_s=1.3)
    replacement = subject.tick(now_s=1.31, safety=SafetySignal())
    assert replacement.decision is ScheduleDecision.EXECUTE
    assert replacement.action[0] > first.action[0]
    assert replacement.action[0] <= first.action[0] + 0.1


def test_contact_and_tracker_veto_cancel_remaining_chunk() -> None:
    for signal, expected in [
        (SafetySignal(contact=True), "contact_detected"),
        (SafetySignal(tracker_veto=True), "tracker_veto"),
    ]:
        subject = scheduler()
        subject.submit([[0.1, 0.0], [0.2, 0.0]],
                       observation_time_s=2.0, inference_completed_time_s=2.1, now_s=2.1)
        result = subject.tick(now_s=2.11, safety=signal)
        assert result.decision is ScheduleDecision.CANCEL
        assert result.reason == expected
        assert subject.tick(now_s=2.12, safety=SafetySignal()).decision is ScheduleDecision.HOLD


def test_clearance_and_confidence_are_fail_closed() -> None:
    subject = scheduler()
    subject.submit([[0.1, 0.0]], observation_time_s=3.0,
                   inference_completed_time_s=3.1, now_s=3.1)
    result = subject.tick(now_s=3.11, safety=SafetySignal(table_clearance_m=0.049))
    assert result.reason == "table_clearance_low"

    subject = scheduler()
    subject.submit([[0.1, 0.0]], observation_time_s=3.0,
                   inference_completed_time_s=3.1, now_s=3.1)
    result = subject.tick(now_s=3.11, safety=SafetySignal(tracker_confidence=0.44))
    assert result.reason == "tracker_confidence_low"


def test_out_of_order_chunk_cannot_overwrite_newer_policy_output() -> None:
    subject = scheduler()
    assert subject.submit([[0.1, 0.0]], observation_time_s=4.0,
                          inference_completed_time_s=4.1, now_s=4.1).decision is ScheduleDecision.EXECUTE
    rejected = subject.submit([[0.2, 0.0]], observation_time_s=3.9,
                              inference_completed_time_s=4.11, now_s=4.11)
    assert rejected.reason == "out_of_order_observation"
