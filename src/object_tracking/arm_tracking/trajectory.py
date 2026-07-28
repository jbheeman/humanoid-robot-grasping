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
