"""Physics-qualified metrics and promotion gates for bunny-block rollouts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping


REQUIRED_POLICIES = (
    "hold",
    "ground_truth",
    "scripted_ik",
    "tracker_cv_ik",
    "vla",
)
REQUIRED_LATENCY_PROFILES = ("p50", "p95", "1.5xp95")


@dataclass(frozen=True)
class BlockSuccessConfig:
    minimum_contact_dwell_s: float = 0.20
    maximum_post_contact_speed_m_s: float = 0.05
    maximum_speed_ratio: float = 0.40
    maximum_post_contact_progress_m: float = 0.03
    maximum_peak_impact_n: float = 20.0


@dataclass(frozen=True)
class PromotionConfig:
    minimum_vla_episodes: int = 300
    minimum_latency_slice_episodes: int = 50
    minimum_vla_success_rate: float = 0.80
    minimum_vla_wilson_lower_95: float = 0.70
    minimum_scripted_success_rate: float = 0.95
    minimum_ground_truth_success_rate: float = 0.98
    maximum_hold_success_rate: float = 0.15
    minimum_success_gain_over_hold: float = 0.20
    minimum_normalized_scripted_gap_closed: float = 0.50
    maximum_safety_intervention_rate: float = 0.02
    maximum_latency_slice_failure_rate: float = 0.30


def physics_qualified_success(
    record: Mapping[str, object],
    config: BlockSuccessConfig | None = None,
) -> bool:
    """Require an actual controlled block, not a one-frame contact label."""

    cfg = config or BlockSuccessConfig()
    pre_speed = float(record["pre_contact_speed_m_s"])
    post_speed = float(record["post_contact_speed_m_s"])
    speed_ratio = post_speed / max(pre_speed, 1e-6)
    slowed = (
        post_speed <= cfg.maximum_post_contact_speed_m_s
        or speed_ratio <= cfg.maximum_speed_ratio
    )
    return bool(
        record["contact"]
        and slowed
        and float(record["post_contact_progress_m"])
        <= cfg.maximum_post_contact_progress_m
        and float(record["contact_dwell_s"]) >= cfg.minimum_contact_dwell_s
        and float(record["peak_impact_n"]) <= cfg.maximum_peak_impact_n
        and int(record["prohibited_contacts"]) == 0
    )


def wilson_lower_95(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def summarize_rollouts(
    records: Iterable[Mapping[str, object]],
    success_config: BlockSuccessConfig | None = None,
) -> dict[str, object]:
    rows = tuple(records)
    if not rows:
        raise ValueError("at least one rollout is required")
    ids: set[tuple[str, str, int]] = set()
    policy_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        policy = str(row["policy"])
        latency_profile = str(row["latency_profile"])
        seed = int(row["seed"])
        key = (policy, latency_profile, seed)
        if key in ids:
            raise ValueError(f"duplicate rollout key: {key}")
        ids.add(key)
        if policy not in REQUIRED_POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        if latency_profile not in REQUIRED_LATENCY_PROFILES:
            raise ValueError(f"unknown latency profile: {latency_profile}")
        policy_rows[policy].append(row)

    policies: dict[str, object] = {}
    for policy in REQUIRED_POLICIES:
        selected = policy_rows.get(policy, [])
        successes = sum(physics_qualified_success(row, success_config) for row in selected)
        total = len(selected)
        latency_slices: dict[str, object] = {}
        for profile in REQUIRED_LATENCY_PROFILES:
            profile_rows = [
                row for row in selected if str(row["latency_profile"]) == profile
            ]
            profile_successes = sum(
                physics_qualified_success(row, success_config) for row in profile_rows
            )
            latency_slices[profile] = {
                "episodes": len(profile_rows),
                "successes": profile_successes,
                "success_rate": (
                    profile_successes / len(profile_rows) if profile_rows else 0.0
                ),
            }
        policies[policy] = {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "wilson_lower_95": wilson_lower_95(successes, total),
            "prohibited_contacts": sum(
                int(row["prohibited_contacts"]) for row in selected
            ),
            "stale_waypoint_executions": sum(
                int(row.get("stale_waypoint_executions", 0)) for row in selected
            ),
            "safety_intervention_rate": (
                sum(bool(row.get("safety_intervention", False)) for row in selected)
                / total
                if total
                else 0.0
            ),
            "latency_profiles": latency_slices,
        }

    rates = {
        policy: float(policies[policy]["success_rate"])  # type: ignore[index]
        for policy in REQUIRED_POLICIES
    }
    denominator = rates["scripted_ik"] - rates["hold"]
    normalized_gap = (
        (rates["vla"] - rates["hold"]) / denominator if denominator > 0 else 0.0
    )
    return {
        "schema_version": 1,
        "success_definition": "physics_qualified_block_v1",
        "policies": policies,
        "vla_success_gain_over_hold": rates["vla"] - rates["hold"],
        "vla_normalized_scripted_gap_closed": normalized_gap,
    }


def evaluate_promotion(
    summary: Mapping[str, object],
    config: PromotionConfig | None = None,
) -> dict[str, object]:
    cfg = config or PromotionConfig()
    policies = summary["policies"]
    assert isinstance(policies, Mapping)
    missing = [name for name in REQUIRED_POLICIES if name not in policies]
    if missing:
        raise ValueError(f"missing required policy rollouts: {', '.join(missing)}")

    vla = policies["vla"]
    ground_truth = policies["ground_truth"]
    scripted = policies["scripted_ik"]
    hold = policies["hold"]
    assert isinstance(vla, Mapping)
    assert isinstance(ground_truth, Mapping)
    assert isinstance(scripted, Mapping)
    assert isinstance(hold, Mapping)

    latency_profiles = vla["latency_profiles"]
    assert isinstance(latency_profiles, Mapping)
    latency_checks = {
        profile: (
            int(latency_profiles[profile]["episodes"])  # type: ignore[index]
            >= cfg.minimum_latency_slice_episodes
            and float(latency_profiles[profile]["success_rate"])  # type: ignore[index]
            >= 1.0 - cfg.maximum_latency_slice_failure_rate
        )
        for profile in REQUIRED_LATENCY_PROFILES
    }
    improvement_check = bool(
        float(summary["vla_success_gain_over_hold"])
        >= cfg.minimum_success_gain_over_hold
        or float(summary["vla_normalized_scripted_gap_closed"])
        >= cfg.minimum_normalized_scripted_gap_closed
    )
    checks = {
        "enough_vla_rollouts": int(vla["episodes"]) >= cfg.minimum_vla_episodes,
        "ground_truth_replay_valid": (
            float(ground_truth["success_rate"])
            >= cfg.minimum_ground_truth_success_rate
            and int(ground_truth["prohibited_contacts"]) == 0
        ),
        "scripted_ik_ceiling_valid": (
            float(scripted["success_rate"]) >= cfg.minimum_scripted_success_rate
            and int(scripted["prohibited_contacts"]) == 0
        ),
        "hold_baseline_valid": (
            float(hold["success_rate"]) <= cfg.maximum_hold_success_rate
        ),
        "vla_success_rate": (
            float(vla["success_rate"]) >= cfg.minimum_vla_success_rate
        ),
        "vla_confidence_bound": (
            float(vla["wilson_lower_95"]) >= cfg.minimum_vla_wilson_lower_95
        ),
        "vla_improves_over_hold": improvement_check,
        "zero_vla_prohibited_contacts": int(vla["prohibited_contacts"]) == 0,
        "zero_stale_waypoint_executions": (
            int(vla["stale_waypoint_executions"]) == 0
        ),
        "bounded_safety_interventions": (
            float(vla["safety_intervention_rate"])
            <= cfg.maximum_safety_intervention_rate
        ),
        **{
            f"latency_{profile}": passed
            for profile, passed in latency_checks.items()
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "robot_execution_authorized": False,
    }
