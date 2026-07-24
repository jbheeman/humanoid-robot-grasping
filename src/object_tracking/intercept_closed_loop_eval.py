"""Generic metrics and gates for lightweight bunny-interception policies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .vla_closed_loop_eval import BlockSuccessConfig, physics_qualified_success, wilson_lower_95


LIGHTWEIGHT_POLICIES = ("hold", "oracle_ik", "cv_ik", "alpha_beta_ik", "gru_ik")
LATENCY_SLICES_MS = (0, 100, 200, 400)


@dataclass(frozen=True)
class InterceptPromotionConfig:
    minimum_scenarios_per_slice: int = 30
    minimum_oracle_success_rate: float = 29.0 / 30.0
    minimum_candidate_success_rate: float = 0.80
    minimum_candidate_wilson_lower_95: float = 0.70
    maximum_hold_success_rate: float = 0.15


def summarize_intercept_rollouts(
    records: Iterable[Mapping[str, object]],
    *,
    success_config: BlockSuccessConfig | None = None,
) -> dict[str, object]:
    rows = tuple(records)
    if not rows:
        raise ValueError("at least one rollout is required")
    keys: set[tuple[str, int, str]] = set()
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    scenario_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        policy = str(row["policy"])
        latency_ms = int(row["latency_ms"])
        scenario_id = str(row["scenario_id"])
        key = (policy, latency_ms, scenario_id)
        if key in keys:
            raise ValueError(f"duplicate rollout key: {key}")
        if policy not in LIGHTWEIGHT_POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        if latency_ms not in LATENCY_SLICES_MS:
            raise ValueError(f"unsupported injected latency: {latency_ms} ms")
        keys.add(key)
        grouped[policy].append(row)
        scenario_sets[(policy, latency_ms)].add(scenario_id)

    reference = scenario_sets.get(("oracle_ik", 0), set())
    if not reference:
        raise ValueError("oracle_ik at zero latency is required")
    unmatched = {
        f"{policy}@{latency_ms}": sorted(reference.symmetric_difference(scenarios))
        for (policy, latency_ms), scenarios in scenario_sets.items()
        if policy != "hold" and scenarios != reference
    }

    policies: dict[str, object] = {}
    for policy in LIGHTWEIGHT_POLICIES:
        selected = grouped.get(policy, [])
        successes = sum(physics_qualified_success(row, success_config) for row in selected)
        failure_reasons: dict[str, int] = defaultdict(int)
        for row in selected:
            if not physics_qualified_success(row, success_config):
                failure_reasons[str(row.get("failure_reason", "physics_block_failed"))] += 1
        slices: dict[str, object] = {}
        for latency_ms in LATENCY_SLICES_MS:
            subset = [row for row in selected if int(row["latency_ms"]) == latency_ms]
            slice_successes = sum(
                physics_qualified_success(row, success_config) for row in subset
            )
            slices[str(latency_ms)] = {
                "episodes": len(subset),
                "successes": slice_successes,
                "success_rate": slice_successes / len(subset) if subset else 0.0,
                "mean_prediction_error_m": (
                    sum(float(row["prediction_error_m"]) for row in subset)
                    / len(subset)
                    if subset and all(row.get("prediction_error_m") is not None for row in subset)
                    else None
                ),
            }
        policies[policy] = {
            "episodes": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected) if selected else 0.0,
            "wilson_lower_95": wilson_lower_95(successes, len(selected)),
            "prohibited_contacts": sum(int(row["prohibited_contacts"]) for row in selected),
            "deadline_unreachable": sum(
                str(row.get("failure_reason", "")) == "deadline_unreachable"
                for row in selected
            ),
            "ik_rejections": sum(bool(row.get("ik_rejected", False)) for row in selected),
            "joint_limit_saturations": sum(
                int(row.get("joint_limit_saturations", 0)) for row in selected
            ),
            "torque_saturations": sum(
                int(row.get("torque_saturations", 0)) for row in selected
            ),
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "latency_slices_ms": slices,
        }
    return {
        "schema_version": 1,
        "success_definition": "physics_qualified_block_v1",
        "latency_slices_ms": list(LATENCY_SLICES_MS),
        "paired_scenarios": not unmatched,
        "unmatched_scenarios": unmatched,
        "policies": policies,
    }


def evaluate_intercept_promotion(
    summary: Mapping[str, object],
    *,
    candidate_policy: str,
    config: InterceptPromotionConfig | None = None,
) -> dict[str, object]:
    if candidate_policy not in ("cv_ik", "alpha_beta_ik", "gru_ik"):
        raise ValueError("candidate_policy must be a lightweight prediction policy")
    cfg = config or InterceptPromotionConfig()
    policies = summary["policies"]
    assert isinstance(policies, Mapping)
    oracle = policies["oracle_ik"]
    hold = policies["hold"]
    candidate = policies[candidate_policy]
    assert isinstance(oracle, Mapping)
    assert isinstance(hold, Mapping)
    assert isinstance(candidate, Mapping)
    oracle_zero = oracle["latency_slices_ms"]["0"]  # type: ignore[index]
    candidate_slices = candidate["latency_slices_ms"]
    assert isinstance(oracle_zero, Mapping)
    assert isinstance(candidate_slices, Mapping)

    candidate_slice_checks = {
        latency: (
            int(candidate_slices[str(latency)]["episodes"])  # type: ignore[index]
            >= cfg.minimum_scenarios_per_slice
            and float(candidate_slices[str(latency)]["success_rate"])  # type: ignore[index]
            >= cfg.minimum_candidate_success_rate
        )
        for latency in (100, 200, 400)
    }
    checks = {
        "paired_scenarios": bool(summary["paired_scenarios"]),
        "oracle_control_valid": (
            int(oracle_zero["episodes"]) >= cfg.minimum_scenarios_per_slice
            and float(oracle_zero["success_rate"]) >= cfg.minimum_oracle_success_rate
            and int(oracle["prohibited_contacts"]) == 0
        ),
        "hold_baseline_valid": (
            float(hold["success_rate"]) <= cfg.maximum_hold_success_rate
        ),
        "candidate_success_rate": (
            float(candidate["success_rate"]) >= cfg.minimum_candidate_success_rate
        ),
        "candidate_confidence_bound": (
            float(candidate["wilson_lower_95"]) >= cfg.minimum_candidate_wilson_lower_95
        ),
        "zero_prohibited_contacts": int(candidate["prohibited_contacts"]) == 0,
        "zero_joint_limit_saturations": int(candidate["joint_limit_saturations"]) == 0,
        "zero_torque_saturations": int(candidate["torque_saturations"]) == 0,
        **{f"latency_{latency}ms": passed for latency, passed in candidate_slice_checks.items()},
    }
    return {
        "candidate_policy": candidate_policy,
        "passed": all(checks.values()),
        "checks": checks,
        "robot_execution_authorized": False,
    }
