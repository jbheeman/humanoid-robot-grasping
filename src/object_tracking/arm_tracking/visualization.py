"""Read-only, versioned full-body visualization state composition."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

from object_tracking._compat import strict_zip

from .joints import ARM_INDICES, BODY_JOINT_NAMES, BODY_MODEL_ID


VISUALIZATION_SCHEMA_VERSION = 1


def _pose(values: Sequence[float] | None) -> list[float] | None:
    if values is None or len(values) != 29:
        return None
    pose = [float(value) for value in values]
    return pose if all(math.isfinite(value) for value in pose) else None


def compose_commanded_body_pose(
    measured_body_q: Sequence[float] | None,
    commanded_arm_q: Sequence[float] | None,
) -> list[float] | None:
    """Keep measured legs/waist and replace only the 14 commanded arm slots."""

    result = _pose(measured_body_q)
    if result is None:
        return None
    if commanded_arm_q is None:
        return result
    if len(commanded_arm_q) != len(ARM_INDICES):
        raise ValueError("commanded_arm_q must contain all 14 arm joints")
    arm = [float(value) for value in commanded_arm_q]
    if not all(math.isfinite(value) for value in arm):
        raise ValueError("commanded_arm_q must contain finite values")
    for index, value in strict_zip(ARM_INDICES, arm):
        result[index] = value
    return result


def visualization_state(
    *,
    measured_body_q: Sequence[float] | None,
    measured_body_dq: Sequence[float] | None = None,
    commanded_arm_q: Sequence[float] | None = None,
    commanded_body_q: Sequence[float] | None = None,
    reference_body_q: Sequence[float] | None = None,
    received_at: float | None = None,
    now: float | None = None,
    state_ttl_s: float = 0.25,
    selected_joint: str | None = None,
    faulted_joints: Sequence[str] = (),
    available: bool | None = None,
) -> dict[str, Any]:
    """Build the stable browser contract without creating a control capability."""

    current = time.monotonic() if now is None else now
    measured = _pose(measured_body_q)
    reference = _pose(reference_body_q) or (None if measured is None else list(measured))
    if commanded_body_q is not None:
        commanded = _pose(commanded_body_q)
    else:
        commanded = compose_commanded_body_pose(measured, commanded_arm_q)
    velocities = _pose(measured_body_dq)
    age_ms = None if received_at is None else round(max(0.0, current - received_at) * 1000.0, 3)
    is_available = measured is not None if available is None else bool(available and measured is not None)
    fresh = bool(is_available and age_ms is not None and age_ms <= state_ttl_s * 1000.0)
    return {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "read_only": True,
        "model_id": BODY_MODEL_ID,
        "joint_names": list(BODY_JOINT_NAMES),
        "reference_pose_rad": reference,
        "measured_pose_rad": measured,
        "measured_velocity_rad_s": velocities,
        "commanded_pose_rad": commanded,
        "received_monotonic_s": received_at,
        "state_age_ms": age_ms,
        "fresh": fresh,
        "available": is_available,
        "selected_joint": selected_joint if selected_joint in BODY_JOINT_NAMES else None,
        "faulted_joints": [name for name in faulted_joints if name in BODY_JOINT_NAMES],
    }
