"""Ruckig trajectory evidence shared by live planning and simulation."""

from __future__ import annotations

import math
from typing import Sequence

from ruckig import InputParameter, Result, Ruckig, Trajectory


_DOF = 7


def _state(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != _DOF or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {_DOF} finite values")
    return result


def _positive_limits(values: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        result = (float(values),) * _DOF
    else:
        result = _state(values, name)
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain positive values")
    return result


def minimum_ruckig_duration_s(
    *,
    current_position: Sequence[float],
    target_position: Sequence[float],
    maximum_velocity: float | Sequence[float],
    maximum_acceleration: float | Sequence[float],
    maximum_jerk: float | Sequence[float],
    current_velocity: Sequence[float] = (0.0,) * _DOF,
    current_acceleration: Sequence[float] = (0.0,) * _DOF,
    target_velocity: Sequence[float] = (0.0,) * _DOF,
    target_acceleration: Sequence[float] = (0.0,) * _DOF,
) -> float:
    """Return the time-optimal synchronized duration for one arm target.

    This performs an offline Ruckig calculation only. It does not publish or
    open any robot transport, which makes it safe for planner and simulator
    reachability checks.
    """

    inp = InputParameter(_DOF)
    inp.current_position = _state(current_position, "current_position")
    inp.current_velocity = _state(current_velocity, "current_velocity")
    inp.current_acceleration = _state(current_acceleration, "current_acceleration")
    inp.target_position = _state(target_position, "target_position")
    inp.target_velocity = _state(target_velocity, "target_velocity")
    inp.target_acceleration = _state(target_acceleration, "target_acceleration")
    inp.max_velocity = _positive_limits(maximum_velocity, "maximum_velocity")
    inp.max_acceleration = _positive_limits(
        maximum_acceleration, "maximum_acceleration"
    )
    inp.max_jerk = _positive_limits(maximum_jerk, "maximum_jerk")

    trajectory = Trajectory(_DOF)
    result = Ruckig(_DOF).calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise ValueError(f"Ruckig could not calculate a trajectory: {result}")
    duration = float(trajectory.duration)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("Ruckig returned an invalid trajectory duration")
    return duration


def ruckig_position_samples(
    *,
    current_position: Sequence[float],
    target_position: Sequence[float],
    maximum_velocity: float | Sequence[float],
    maximum_acceleration: float | Sequence[float],
    maximum_jerk: float | Sequence[float],
    sample_period_s: float = 0.004,
    current_velocity: Sequence[float] = (0.0,) * _DOF,
    current_acceleration: Sequence[float] = (0.0,) * _DOF,
) -> tuple[tuple[float, ...], ...]:
    """Sample the synchronized zero-endpoint-velocity trajectory for one edge."""

    if not math.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    inp = InputParameter(_DOF)
    inp.current_position = _state(current_position, "current_position")
    inp.current_velocity = _state(current_velocity, "current_velocity")
    inp.current_acceleration = _state(current_acceleration, "current_acceleration")
    inp.target_position = _state(target_position, "target_position")
    inp.target_velocity = (0.0,) * _DOF
    inp.target_acceleration = (0.0,) * _DOF
    inp.max_velocity = _positive_limits(maximum_velocity, "maximum_velocity")
    inp.max_acceleration = _positive_limits(maximum_acceleration, "maximum_acceleration")
    inp.max_jerk = _positive_limits(maximum_jerk, "maximum_jerk")
    trajectory = Trajectory(_DOF)
    result = Ruckig(_DOF).calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise ValueError(f"Ruckig could not calculate a trajectory: {result}")
    steps = max(1, int(math.ceil(float(trajectory.duration) / sample_period_s)))
    return tuple(
        tuple(float(value) for value in trajectory.at_time(float(trajectory.duration) * i / steps)[0])
        for i in range(steps + 1)
    )


def minimum_ruckig_path_duration_s(
    *,
    current_position: Sequence[float],
    target_positions: Sequence[Sequence[float]],
    maximum_velocity: float | Sequence[float],
    maximum_acceleration: float | Sequence[float],
    maximum_jerk: float | Sequence[float],
    current_velocity: Sequence[float] = (0.0,) * _DOF,
    current_acceleration: Sequence[float] = (0.0,) * _DOF,
) -> float:
    """Return conservative duration through all remaining joint waypoints.

    Each intermediate collision-checked waypoint is treated as a zero-velocity
    stop. This cannot understate the time needed by the downstream online
    Ruckig bridge, even when the live scheduler later preserves velocity
    through a safe lookahead edge.
    """

    position = _state(current_position, "current_position")
    velocity = _state(current_velocity, "current_velocity")
    acceleration = _state(current_acceleration, "current_acceleration")
    total = 0.0
    for index, target in enumerate(target_positions):
        target_position = _state(target, f"target_positions[{index}]")
        total += minimum_ruckig_duration_s(
            current_position=position,
            current_velocity=velocity,
            current_acceleration=acceleration,
            target_position=target_position,
            maximum_velocity=maximum_velocity,
            maximum_acceleration=maximum_acceleration,
            maximum_jerk=maximum_jerk,
        )
        position = target_position
        velocity = (0.0,) * _DOF
        acceleration = (0.0,) * _DOF
    return total
