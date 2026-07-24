from object_tracking.intercept_closed_loop_eval import (
    InterceptPromotionConfig,
    evaluate_intercept_promotion,
    summarize_intercept_rollouts,
)


def rollout(policy: str, latency_ms: int, scenario: int, success: bool) -> dict:
    return {
        "policy": policy,
        "latency_ms": latency_ms,
        "scenario_id": f"scenario-{scenario:02d}",
        "contact": success,
        "pre_contact_speed_m_s": 0.12,
        "post_contact_speed_m_s": 0.02 if success else 0.10,
        "post_contact_progress_m": 0.01 if success else 0.10,
        "contact_dwell_s": 0.25 if success else 0.0,
        "peak_impact_n": 4.0,
        "prohibited_contacts": 0,
        "prediction_error_m": 0.01,
        "failure_reason": "" if success else "physics_block_failed",
        "ik_rejected": False,
        "joint_limit_saturations": 0,
        "torque_saturations": 0,
    }


def test_lightweight_promotion_uses_paired_numeric_latency_slices() -> None:
    records = []
    for policy in ("hold", "oracle_ik", "cv_ik", "alpha_beta_ik", "gru_ik"):
        for latency_ms in (0, 100, 200, 400):
            for scenario in range(30):
                success = policy != "hold"
                records.append(rollout(policy, latency_ms, scenario, success))
    summary = summarize_intercept_rollouts(records)
    promotion = evaluate_intercept_promotion(
        summary,
        candidate_policy="alpha_beta_ik",
        config=InterceptPromotionConfig(minimum_candidate_wilson_lower_95=0.70),
    )
    assert summary["paired_scenarios"]
    assert promotion["passed"]
    assert not promotion["robot_execution_authorized"]


def test_deadline_failure_is_reported_separately() -> None:
    records = []
    for policy in ("hold", "oracle_ik", "cv_ik", "alpha_beta_ik", "gru_ik"):
        for latency_ms in (0, 100, 200, 400):
            for scenario in range(30):
                success = policy not in ("hold", "gru_ik")
                row = rollout(policy, latency_ms, scenario, success)
                if policy == "gru_ik":
                    row["failure_reason"] = "deadline_unreachable"
                records.append(row)
    summary = summarize_intercept_rollouts(records)
    assert summary["policies"]["gru_ik"]["deadline_unreachable"] == 120
