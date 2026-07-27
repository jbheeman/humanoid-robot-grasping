"""Deterministic geometry and IK gateway for UnifoLM action chunks.

This module has no ROS or Unitree transport imports.  A VLA proposes task
motion; this gateway is the only component allowed to turn those proposals
into joint paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from object_tracking.arm_tracking.geometry import SupportRegion
from object_tracking.unifolm_vla import UnifoLMWaypoint


@dataclass(frozen=True)
class IKGatewayConfig:
    minimum_table_clearance_m: float = 0.05
    maximum_table_projection_m: float = 0.10
    maximum_joint_step_rad: float = 0.025
    maximum_waypoints: int = 5

    def __post_init__(self) -> None:
        values = (
            self.minimum_table_clearance_m,
            self.maximum_table_projection_m,
            self.maximum_joint_step_rad,
        )
        if not all(np.isfinite(values)):
            raise ValueError("IK gateway limits must be finite")
        if self.minimum_table_clearance_m < 0.0:
            raise ValueError("minimum table clearance must be non-negative")
        if self.maximum_table_projection_m <= 0.0:
            raise ValueError("maximum table projection must be positive")
        if not 0.0 < self.maximum_joint_step_rad <= 0.05:
            raise ValueError("maximum joint step must be in (0, 0.05] rad")
        if not 1 <= self.maximum_waypoints <= 25:
            raise ValueError("maximum waypoints must be between 1 and 25")


@dataclass(frozen=True)
class IKGatewayResult:
    ok: bool
    q_path: tuple[tuple[float, ...], ...]
    reason: str | None
    projected_waypoints: int
    maximum_projection_m: float

    @property
    def intervention_fraction(self) -> float:
        if not self.q_path:
            return 0.0
        return self.projected_waypoints / len(self.q_path)


class GeometricIKGateway:
    """Project, solve, and continuously validate a VLA action chunk."""

    def __init__(self, solver: Any, config: IKGatewayConfig | None = None) -> None:
        self.solver = solver
        self.config = config or IKGatewayConfig()

    def plan(
        self,
        waypoints: Sequence[UnifoLMWaypoint],
        measured_q_rad: Sequence[float],
        support: SupportRegion,
    ) -> IKGatewayResult:
        start = np.asarray(measured_q_rad, dtype=float)
        if start.shape != (7,) or not np.all(np.isfinite(start)):
            return IKGatewayResult(False, (), "invalid_measured_joint_state", 0, 0.0)
        if not waypoints:
            return IKGatewayResult(False, (), "empty_action_chunk", 0, 0.0)

        seed = tuple(float(value) for value in start)
        planned: list[tuple[float, ...]] = []
        projected_count = 0
        maximum_projection = 0.0
        for waypoint in waypoints[: self.config.maximum_waypoints]:
            target = waypoint.right_transform()
            original = target[:3, 3].copy()
            projected = support.project_to_clearance(
                original,
                minimum_clearance_m=self.config.minimum_table_clearance_m,
            )
            correction = float(np.linalg.norm(projected - original))
            maximum_projection = max(maximum_projection, correction)
            if correction > self.config.maximum_table_projection_m:
                return IKGatewayResult(
                    False,
                    (),
                    (
                        "table_projection_limit:"
                        f"required={correction:.5f}:"
                        f"limit={self.config.maximum_table_projection_m:.5f}"
                    ),
                    projected_count,
                    maximum_projection,
                )
            projected_count += int(correction > 1e-9)
            target[:3, 3] = projected
            result = self.solver.solve_local_translation(
                target,
                seed,
                support_plane=support,
                maximum_joint_step_rad=self.config.maximum_joint_step_rad,
                validate_path=True,
            )
            if not result.ok or result.q_rad is None:
                return IKGatewayResult(
                    False,
                    (),
                    f"ik_rejected:{result.reason or 'unknown'}",
                    projected_count,
                    maximum_projection,
                )
            candidate = tuple(float(value) for value in result.q_rad)
            if len(candidate) != 7 or not np.all(np.isfinite(candidate)):
                return IKGatewayResult(
                    False,
                    (),
                    "ik_returned_invalid_joint_state",
                    projected_count,
                    maximum_projection,
                )
            seed = candidate
            planned.append(candidate)

        path_error = self.solver.validate_joint_path(
            (tuple(float(value) for value in start), *planned),
            support_plane=support,
        )
        if path_error:
            return IKGatewayResult(
                False,
                (),
                f"swept_path_rejected:{path_error}",
                projected_count,
                maximum_projection,
            )
        return IKGatewayResult(
            True,
            tuple(planned),
            None,
            projected_count,
            maximum_projection,
        )
