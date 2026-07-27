from object_tracking.vla_closed_loop_eval import (
    PromotionConfig,
    evaluate_promotion,
    physics_qualified_success,
    summarize_rollouts,
)


def rollout(policy: str, seed: int, latency: str, success: bool = True) -> dict:
    return {
        "policy": policy,
        "seed": seed,
        "latency_profile": latency,
        "contact": success,
        "pre_contact_speed_m_s": 0.20,
        "post_contact_speed_m_s": 0.02 if success else 0.18,
        "post_contact_progress_m": 0.01 if success else 0.10,
        "contact_dwell_s": 0.30 if success else 0.0,
        "peak_impact_n": 5.0,
        "prohibited_contacts": 0,
        "stale_waypoint_executions": 0,
        "safety_intervention": False,
    }


def test_contact_alone_does_not_count_as_a_block() -> None:
    record = rollout("vla", 1, "p50")
    record["post_contact_speed_m_s"] = 0.19
    record["post_contact_progress_m"] = 0.08
    assert not physics_qualified_success(record)


def test_promotion_requires_baselines_latency_slices_and_safety() -> None:
    records = []
    for policy in ("hold", "ground_truth", "scripted_ik", "tracker_cv_ik", "vla"):
        for profile_index, profile in enumerate(("p50", "p95", "1.5xp95")):
            for index in range(10):
                success = policy != "hold"
                if policy == "tracker_cv_ik" and index >= 7:
                    success = False
                records.append(
                    rollout(policy, profile_index * 100 + index, profile, success)
                )

    summary = summarize_rollouts(records)
    promotion = evaluate_promotion(
        summary,
        PromotionConfig(
            minimum_vla_episodes=30,
            minimum_latency_slice_episodes=10,
            minimum_vla_wilson_lower_95=0.80,
        ),
    )

    assert promotion["passed"]
    assert summary["vla_success_gain_over_hold"] == 1.0
    assert summary["vla_normalized_scripted_gap_closed"] == 1.0
    assert not promotion["robot_execution_authorized"]


def test_ground_truth_replay_failure_blocks_promotion() -> None:
    records = []
    for policy in ("hold", "ground_truth", "scripted_ik", "tracker_cv_ik", "vla"):
        for profile_index, profile in enumerate(("p50", "p95", "1.5xp95")):
            for index in range(10):
                success = policy not in ("hold", "ground_truth")
                records.append(
                    rollout(policy, profile_index * 100 + index, profile, success)
                )
    summary = summarize_rollouts(records)
    promotion = evaluate_promotion(
        summary,
        PromotionConfig(
            minimum_vla_episodes=30,
            minimum_latency_slice_episodes=10,
            minimum_vla_wilson_lower_95=0.80,
        ),
    )
    assert not promotion["passed"]
    assert not promotion["checks"]["ground_truth_replay_valid"]
