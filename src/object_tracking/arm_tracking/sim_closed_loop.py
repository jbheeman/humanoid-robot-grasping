"""Transport-neutral contract for closed-loop G1 arm-core simulation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .geometry import SupportRegion


SCHEMA_VERSION = 1
BODY_DOF = 29
RIGHT_ARM_DOF = 7


def _finite_vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} finite values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite values") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _optional_vector(
    value: object, length: int, name: str
) -> tuple[float, ...] | None:
    return None if value is None else _finite_vector(value, length, name)


def _nonnegative_float(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _required_string(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class ObjectObservation:
    track_id: int
    class_name: str
    confidence: float
    position_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    observation_time_s: float
    consecutive_observations: int
    residual_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        object.__setattr__(self, "class_name", _required_string(self.class_name, "class_name"))
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(
            self, "position_m", _finite_vector(self.position_m, 3, "position_m")
        )
        object.__setattr__(
            self, "velocity_m_s", _finite_vector(self.velocity_m_s, 3, "velocity_m_s")
        )
        object.__setattr__(
            self,
            "observation_time_s",
            _nonnegative_float(self.observation_time_s, "observation_time_s"),
        )
        if self.consecutive_observations < 1:
            raise ValueError("consecutive_observations must be positive")
        object.__setattr__(
            self, "residual_m", _nonnegative_float(self.residual_m, "residual_m")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "position_m": list(self.position_m),
            "velocity_m_s": list(self.velocity_m_s),
            "observation_time_s": self.observation_time_s,
            "consecutive_observations": self.consecutive_observations,
            "residual_m": self.residual_m,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectObservation":
        return cls(
            track_id=value.get("track_id"),
            class_name=value.get("class_name"),
            confidence=float(value.get("confidence")),
            position_m=value.get("position_m"),
            velocity_m_s=value.get("velocity_m_s"),
            observation_time_s=value.get("observation_time_s"),
            consecutive_observations=int(value.get("consecutive_observations")),
            residual_m=float(value.get("residual_m")),
        )


@dataclass(frozen=True)
class SimState:
    """Newest measured state at the post-localization, pre-transport boundary."""

    episode_id: str
    sequence: int
    simulation_time_s: float
    calibration_id: str
    joint_contract_id: str
    body_q_rad: tuple[float, ...]
    body_dq_rad_s: tuple[float, ...]
    support_region: SupportRegion
    object_observation: ObjectObservation | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _required_string(self.episode_id, "episode_id"))
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative_float(self.simulation_time_s, "simulation_time_s"),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _required_string(self.calibration_id, "calibration_id"),
        )
        object.__setattr__(
            self,
            "joint_contract_id",
            _required_string(self.joint_contract_id, "joint_contract_id"),
        )
        object.__setattr__(
            self, "body_q_rad", _finite_vector(self.body_q_rad, BODY_DOF, "body_q_rad")
        )
        object.__setattr__(
            self,
            "body_dq_rad_s",
            _finite_vector(self.body_dq_rad_s, BODY_DOF, "body_dq_rad_s"),
        )
        if (
            self.object_observation is not None
            and self.object_observation.observation_time_s > self.simulation_time_s
        ):
            raise ValueError("object observation cannot be from the future")

    @property
    def right_arm_q_rad(self) -> tuple[float, ...]:
        return self.body_q_rad[22:29]

    @property
    def right_arm_dq_rad_s(self) -> tuple[float, ...]:
        return self.body_dq_rad_s[22:29]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "sim_state",
            "episode_id": self.episode_id,
            "sequence": self.sequence,
            "simulation_time_s": self.simulation_time_s,
            "calibration_id": self.calibration_id,
            "joint_contract_id": self.joint_contract_id,
            "body_q_rad": list(self.body_q_rad),
            "body_dq_rad_s": list(self.body_dq_rad_s),
            "support_region": self.support_region.to_dict(),
            "object_observation": (
                None if self.object_observation is None else self.object_observation.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimState":
        if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != "sim_state":
            raise ValueError("unsupported simulator state contract")
        support = _mapping(value.get("support_region"), "support_region")
        observation = value.get("object_observation")
        return cls(
            episode_id=value.get("episode_id"),
            sequence=value.get("sequence"),
            simulation_time_s=value.get("simulation_time_s"),
            calibration_id=value.get("calibration_id"),
            joint_contract_id=value.get("joint_contract_id"),
            body_q_rad=value.get("body_q_rad"),
            body_dq_rad_s=value.get("body_dq_rad_s"),
            support_region=SupportRegion.from_dict(dict(support)),
            object_observation=(
                None
                if observation is None
                else ObjectObservation.from_dict(
                    _mapping(observation, "object_observation")
                )
            ),
        )


@dataclass(frozen=True)
class SimCommand:
    """Planner result tied to one state and its source observation."""

    episode_id: str
    state_sequence: int
    simulation_time_s: float
    source_observation_time_s: float | None
    status: str
    reason: str
    right_arm_q_rad: tuple[float, ...] | None = None
    right_arm_tau_ff_nm: tuple[float, ...] | None = None
    target_palm_position_m: tuple[float, ...] | None = None
    predicted_crossing_m: tuple[float, ...] | None = None
    crossing_time_from_now_s: float | None = None
    remaining_ruckig_duration_s: float | None = None
    arrival_slack_s: float | None = None
    planning_latency_ms: float = 0.0
    ik_step_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _required_string(self.episode_id, "episode_id"))
        object.__setattr__(
            self, "state_sequence", _nonnegative_int(self.state_sequence, "state_sequence")
        )
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative_float(self.simulation_time_s, "simulation_time_s"),
        )
        if self.status not in ("target", "preview", "hold", "rejected"):
            raise ValueError("status is invalid")
        if not self.reason:
            raise ValueError("reason must not be empty")
        if self.status == "target":
            if self.right_arm_q_rad is None or self.right_arm_tau_ff_nm is None:
                raise ValueError("target commands require q and torque")
        elif self.right_arm_q_rad is not None or self.right_arm_tau_ff_nm is not None:
            raise ValueError("non-target commands cannot contain joint commands")
        object.__setattr__(
            self,
            "right_arm_q_rad",
            _optional_vector(self.right_arm_q_rad, RIGHT_ARM_DOF, "right_arm_q_rad"),
        )
        object.__setattr__(
            self,
            "right_arm_tau_ff_nm",
            _optional_vector(
                self.right_arm_tau_ff_nm, RIGHT_ARM_DOF, "right_arm_tau_ff_nm"
            ),
        )
        object.__setattr__(
            self,
            "target_palm_position_m",
            _optional_vector(
                self.target_palm_position_m, 3, "target_palm_position_m"
            ),
        )
        object.__setattr__(
            self,
            "predicted_crossing_m",
            _optional_vector(self.predicted_crossing_m, 3, "predicted_crossing_m"),
        )
        for name in (
            "source_observation_time_s",
            "crossing_time_from_now_s",
            "remaining_ruckig_duration_s",
            "arrival_slack_s",
            "planning_latency_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_float(value, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "sim_command",
            "episode_id": self.episode_id,
            "state_sequence": self.state_sequence,
            "simulation_time_s": self.simulation_time_s,
            "source_observation_time_s": self.source_observation_time_s,
            "status": self.status,
            "reason": self.reason,
            "right_arm_q_rad": self.right_arm_q_rad,
            "right_arm_tau_ff_nm": self.right_arm_tau_ff_nm,
            "target_palm_position_m": self.target_palm_position_m,
            "predicted_crossing_m": self.predicted_crossing_m,
            "crossing_time_from_now_s": self.crossing_time_from_now_s,
            "remaining_ruckig_duration_s": self.remaining_ruckig_duration_s,
            "arrival_slack_s": self.arrival_slack_s,
            "planning_latency_ms": self.planning_latency_ms,
            "ik_step_type": self.ik_step_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimCommand":
        if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != "sim_command":
            raise ValueError("unsupported simulator command contract")
        return cls(
            episode_id=value.get("episode_id"),
            state_sequence=value.get("state_sequence"),
            simulation_time_s=value.get("simulation_time_s"),
            source_observation_time_s=value.get("source_observation_time_s"),
            status=str(value.get("status", "")),
            reason=str(value.get("reason", "")),
            right_arm_q_rad=value.get("right_arm_q_rad"),
            right_arm_tau_ff_nm=value.get("right_arm_tau_ff_nm"),
            target_palm_position_m=value.get("target_palm_position_m"),
            predicted_crossing_m=value.get("predicted_crossing_m"),
            crossing_time_from_now_s=value.get("crossing_time_from_now_s"),
            remaining_ruckig_duration_s=value.get("remaining_ruckig_duration_s"),
            arrival_slack_s=value.get("arrival_slack_s"),
            planning_latency_ms=value.get("planning_latency_ms", 0.0),
            ik_step_type=value.get("ik_step_type"),
        )


def encode_message(message: SimState | SimCommand) -> bytes:
    payload = json.dumps(
        message.to_dict(), separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    if len(payload) > 64 * 1024:
        raise ValueError("simulator message exceeds 64 KiB")
    return payload + b"\n"


def decode_message(payload: bytes) -> SimState | SimCommand:
    if len(payload) > 64 * 1024:
        raise ValueError("simulator message exceeds 64 KiB")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid simulator JSON message") from exc
    mapping = _mapping(value, "message")
    if mapping.get("kind") == "sim_state":
        return SimState.from_dict(mapping)
    if mapping.get("kind") == "sim_command":
        return SimCommand.from_dict(mapping)
    raise ValueError("unknown simulator message kind")


@dataclass
class SequenceGate:
    """Latch episode identity and discard stale/out-of-order state."""

    episode_id: str | None = None
    calibration_id: str | None = None
    joint_contract_id: str | None = None
    support_fingerprint: str | None = None
    last_sequence: int = -1
    last_simulation_time_s: float = -1.0
    accepted: int = 0
    discarded: int = 0

    def accept(self, state: SimState) -> bool:
        support_fingerprint = json.dumps(
            state.support_region.to_dict(), sort_keys=True, separators=(",", ":")
        )
        identity = (
            state.episode_id,
            state.calibration_id,
            state.joint_contract_id,
            support_fingerprint,
        )
        latched = (
            self.episode_id,
            self.calibration_id,
            self.joint_contract_id,
            self.support_fingerprint,
        )
        if self.episode_id is None:
            (
                self.episode_id,
                self.calibration_id,
                self.joint_contract_id,
                self.support_fingerprint,
            ) = identity
        elif identity != latched:
            raise ValueError("simulator episode identity changed")
        if (
            state.sequence <= self.last_sequence
            or state.simulation_time_s <= self.last_simulation_time_s
        ):
            self.discarded += 1
            return False
        self.last_sequence = state.sequence
        self.last_simulation_time_s = state.simulation_time_s
        self.accepted += 1
        return True
