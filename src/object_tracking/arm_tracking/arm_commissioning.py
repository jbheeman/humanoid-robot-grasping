"""Deterministic right-arm commissioning layered over the persistent bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable

from .arm_bridge import ArmBridgeController, ArmBridgeError, ArmState
from .joints import (
    DEFAULT_RIGHT_JOINT_LIMITS,
    RIGHT_ARM_JOINT_NAMES,
    joint_contract,
    joint_contract_id,
)
from .visualization import visualization_state


# Retained only as a compatibility export for older clients. The explicit
# movement-capable launch command is the acknowledgement; the browser has no
# second text gate.
OPERATOR_ACK = "I HAVE CLEARED THE ROBOT AREA"


class CommissioningPhase(str, Enum):
    CREATED = "CREATED"
    ARMING = "ARMING"
    READY = "READY"
    MOVING = "MOVING"
    AWAITING_CONFIRM = "AWAITING_CONFIRM"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class CommissioningConfig:
    jog_step_rad: float = 0.01
    stage_limit_rad: float = 0.05
    session_limit_rad: float = 0.30
    settle_velocity_rad_s: float = 0.02
    settle_error_rad: float = 0.01
    settle_dwell_s: float = 0.5
    motion_timeout_s: float = 5.0
    nonselected_drift_rad: float = 0.01
    movable_joint_names: tuple[str, ...] = RIGHT_ARM_JOINT_NAMES
    research_root: Path = Path("runs/research/arm_commissioning")
    profile_path: Path = Path.home() / ".config/g1-grasping/right-arm-home.json"


@dataclass
class PendingMotion:
    joint_index: int
    target_q: tuple[float, ...]
    started_at: float
    baseline_q: tuple[float, ...]
    kind: str
    direction: int
    settle_started_at: float | None = None
    peak_error_rad: float = 0.0


@dataclass
class CommissioningSession:
    session_id: str
    client_id: str
    operator: str
    created_at: str
    phase: CommissioningPhase = CommissioningPhase.CREATED
    baseline_q: tuple[float, ...] | None = None
    stage_baseline_q: tuple[float, ...] | None = None
    last_confirmed_q: tuple[float, ...] | None = None
    last_client_sequence: int = -1
    bridge_sequence: int = 0
    pending: PendingMotion | None = None
    sign_checks: dict[str, str] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    candidate: dict[str, Any] | None = None
    replay_validations: int = 0
    fault_reason: str | None = None
    reference_body_q: tuple[float, ...] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def load_home_profile(path: str | Path, *, expected_robot_id: str | None = None) -> dict[str, Any]:
    profile_path = Path(path)
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmBridgeError(
            "Could not read right-arm home profile", code="invalid_home_profile"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ArmBridgeError(
            "Unsupported right-arm home profile schema", code="invalid_home_profile"
        )
    supplied_hash = value.get("profile_hash")
    unhashed = {key: item for key, item in value.items() if key != "profile_hash"}
    encoded = json.dumps(unhashed, separators=(",", ":"), sort_keys=True).encode()
    if not isinstance(supplied_hash, str) or supplied_hash != hashlib.sha256(encoded).hexdigest():
        raise ArmBridgeError("Right-arm home profile hash mismatch", code="invalid_home_profile")
    if value.get("joint_contract_id") != joint_contract_id():
        raise ArmBridgeError("Right-arm home joint contract mismatch", code="invalid_home_profile")
    if value.get("joint_names") != list(RIGHT_ARM_JOINT_NAMES):
        raise ArmBridgeError("Right-arm home joint order mismatch", code="invalid_home_profile")
    if expected_robot_id is not None and (
        not isinstance(value.get("robot"), dict)
        or value["robot"].get("robot_id") != expected_robot_id
    ):
        raise ArmBridgeError("Right-arm home robot identity mismatch", code="invalid_home_profile")
    measured = value.get("measured_q")
    if not isinstance(measured, list) or len(measured) != 7:
        raise ArmBridgeError(
            "Right-arm home must contain seven joints", code="invalid_home_profile"
        )
    try:
        finite = all(math.isfinite(float(item)) for item in measured)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ArmBridgeError(
            "Right-arm home contains non-finite values", code="invalid_home_profile"
        )
    for position, limits in zip(
        (float(item) for item in measured), DEFAULT_RIGHT_JOINT_LIMITS, strict=True
    ):
        if not limits[0] + 0.05 <= position <= limits[1] - 0.05:
            raise ArmBridgeError(
                "Right-arm home violates joint limit margin", code="invalid_home_profile"
            )
    return value


class CommissioningController:
    """One active commissioning session with no queued or arbitrary joint commands."""

    def __init__(
        self,
        bridge: ArmBridgeController,
        config: CommissioningConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        robot_identity: dict[str, str] | None = None,
    ) -> None:
        self.bridge = bridge
        self.config = config or CommissioningConfig()
        self.monotonic = monotonic
        self.robot_identity = robot_identity or {}
        self.session: CommissioningSession | None = None
        self._lock = threading.RLock()
        self._event_path: Path | None = None
        self._telemetry_path: Path | None = None
        self._summary_path: Path | None = None
        self._event_counts: dict[str, int] = {}
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            daemon=True,
            name="arm-commissioning-telemetry",
        )
        self._telemetry_thread.start()

    def create_session(
        self, *, operator_ack: object, operator: object, client_id: object
    ) -> dict[str, Any]:
        with self._lock:
            if self.session is not None and self.session.phase not in {
                CommissioningPhase.STOPPED,
                CommissioningPhase.FAULT,
            }:
                raise ArmBridgeError(
                    "A commissioning session is already active", code="session_conflict"
                )
            operator_name = self._required_text(operator, "operator")
            client = self._required_text(client_id, "client_id")
            session_id = secrets.token_urlsafe(18)
            reference_state = self.bridge.hardware.latest_state()
            session = CommissioningSession(
                session_id=session_id,
                client_id=client,
                operator=operator_name,
                created_at=_utc_now(),
                reference_body_q=(
                    None
                    if reference_state is None
                    else tuple(reference_state.body_q)
                ),
            )
            directory = self.config.research_root / session_id
            directory.mkdir(parents=True, exist_ok=False)
            self._event_path = directory / "events.jsonl"
            self._telemetry_path = directory / "telemetry.jsonl"
            self._summary_path = directory / "summary.json"
            self._event_counts = {}
            manifest = {
                "schema_version": 1,
                "session_id": session_id,
                "created_at": session.created_at,
                "operator": operator_name,
                "client_id": client,
                "joint_contract_id": joint_contract_id(),
                "joint_contract": joint_contract(),
                "robot": self.robot_identity,
                "limits": {
                    "jog_step_rad": self.config.jog_step_rad,
                    "stage_limit_rad": self.config.stage_limit_rad,
                    "session_limit_rad_per_joint": self.config.session_limit_rad,
                },
                "movable_joint_names": list(self.config.movable_joint_names),
            }
            _atomic_json(directory / "manifest.json", manifest)
            self.session = session
            self._event("session_created", {})
            return self.report()

    def enable(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            if session.phase is not CommissioningPhase.CREATED:
                raise ArmBridgeError(
                    "Session cannot be enabled from its current phase", code="invalid_state"
                )
            robot = self._robot_state()
            session.baseline_q = tuple(robot.arm_q[7:])
            session.stage_baseline_q = session.baseline_q
            session.last_confirmed_q = session.baseline_q
            self.bridge.enable_commissioning(
                session_id=session.session_id,
                joint_contract_id=joint_contract_id(),
            )
            session.phase = CommissioningPhase.ARMING
            self._event("enable_requested", {"baseline_q": list(session.baseline_q)})
            return self.report()

    def heartbeat(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self.bridge.heartbeat(session_id=session.session_id)
            return self.report()

    def jog(
        self,
        session_id: str,
        *,
        sequence: object,
        joint_name: object,
        direction: object,
        kind: str = "jog",
    ) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            if session.phase is not CommissioningPhase.READY:
                raise ArmBridgeError(
                    "Wait for the current motion and confirmation", code="motion_busy"
                )
            client_sequence = self._sequence(session, sequence)
            name = self._required_text(joint_name, "joint_name")
            if name not in RIGHT_ARM_JOINT_NAMES:
                raise ArmBridgeError("Unknown right-arm joint", code="unknown_joint")
            if name not in self.config.movable_joint_names:
                raise ArmBridgeError(
                    "Only shoulder joints are enabled for this commissioning run",
                    code="joint_not_enabled",
                )
            if isinstance(direction, bool) or direction not in (-1, 1):
                raise ArmBridgeError("direction must be -1 or 1", code="invalid_direction")
            if kind not in {"jog", "sign_check", "replay"}:
                raise ArmBridgeError("Unknown commissioning motion kind", code="invalid_request")
            robot = self._robot_state()
            measured = tuple(robot.arm_q[7:])
            assert session.baseline_q is not None and session.stage_baseline_q is not None
            index = RIGHT_ARM_JOINT_NAMES.index(name)
            target = list(measured)
            target[index] += int(direction) * self.config.jog_step_rad
            if (
                abs(target[index] - session.stage_baseline_q[index])
                > self.config.stage_limit_rad + 1e-9
            ):
                raise ArmBridgeError(
                    "Approve a new 0.05 rad stage before continuing", code="stage_limit"
                )
            if (
                abs(target[index] - session.baseline_q[index])
                > self.config.session_limit_rad + 1e-9
            ):
                raise ArmBridgeError(
                    "Per-joint commissioning session envelope exceeded", code="session_limit"
                )
            session.bridge_sequence += 1
            self.bridge.set_commissioning_target(
                session_id=session.session_id,
                sequence=session.bridge_sequence,
                right_arm_q=target,
            )
            session.last_client_sequence = client_sequence
            session.pending = PendingMotion(
                joint_index=index,
                target_q=tuple(target),
                started_at=self.monotonic(),
                baseline_q=measured,
                kind=kind,
                direction=int(direction),
            )
            session.phase = CommissioningPhase.MOVING
            self._event(
                "motion_accepted",
                {
                    "sequence": client_sequence,
                    "joint_name": name,
                    "direction": direction,
                    "kind": kind,
                    "measured_start_q": list(measured),
                    "target_q": target,
                },
            )
            return self.report()

    def confirm_motion(
        self, session_id: str, *, sequence: object, outcome: object, notes: object = ""
    ) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            if session.phase is not CommissioningPhase.AWAITING_CONFIRM or session.pending is None:
                raise ArmBridgeError("No settled motion awaits confirmation", code="invalid_state")
            client_sequence = self._sequence(session, sequence)
            result = self._required_text(outcome, "outcome")
            if result not in {
                "expected",
                "reversed",
                "no_motion",
                "unexpected",
                "accepted",
                "rejected",
            }:
                raise ArmBridgeError("Unknown confirmation outcome", code="invalid_outcome")
            pending = session.pending
            robot = self._robot_state()
            measured = tuple(robot.arm_q[7:])
            session.last_client_sequence = client_sequence
            if pending.kind == "sign_check" and result == "expected":
                measured_delta = (
                    measured[pending.joint_index] - pending.baseline_q[pending.joint_index]
                )
                if (
                    measured_delta * pending.direction <= 0.0
                    or not 0.005 <= abs(measured_delta) <= 0.015
                ):
                    raise ArmBridgeError(
                        "Encoder sign-check delta did not match the commanded 0.01 rad motion",
                        code="sign_measurement_mismatch",
                    )
                key = f"{RIGHT_ARM_JOINT_NAMES[pending.joint_index]}:{pending.direction:+d}"
                session.sign_checks[key] = "passed"
            if result in {"expected", "accepted"}:
                session.last_confirmed_q = measured
                if session.candidate is not None:
                    candidate_q = tuple(float(value) for value in session.candidate["measured_q"])
                    departure = max(
                        abs(actual - target)
                        for actual, target in zip(measured, candidate_q, strict=True)
                    )
                    session.candidate["max_departure_rad"] = max(
                        float(session.candidate.get("max_departure_rad", 0.0)), departure
                    )
                    if departure >= 0.02:
                        session.candidate["departed"] = True
            self._event(
                "motion_confirmed",
                {
                    "sequence": client_sequence,
                    "outcome": result,
                    "notes": str(notes)[:500],
                    "kind": pending.kind,
                    "measured_end_q": list(measured),
                    "peak_error_rad": pending.peak_error_rad,
                },
            )
            session.pending = None
            session.phase = CommissioningPhase.READY
            return self.report()

    def checkpoint(self, session_id: str, *, label: object) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            if session.phase is not CommissioningPhase.READY:
                raise ArmBridgeError(
                    "Checkpoint requires a settled ready arm", code="invalid_state"
                )
            measured = tuple(self._robot_state().arm_q[7:])
            checkpoint = {
                "checkpoint_id": secrets.token_urlsafe(8),
                "label": self._required_text(label, "label"),
                "created_at": _utc_now(),
                "measured_q": list(measured),
            }
            session.stage_baseline_q = measured
            session.last_confirmed_q = measured
            session.checkpoints.append(checkpoint)
            self._event("checkpoint_created", checkpoint)
            return self.report()

    def capture_candidate(self, session_id: str, *, label: object) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            required = {
                f"{name}:{direction:+d}" for name in RIGHT_ARM_JOINT_NAMES for direction in (-1, 1)
            }
            if set(session.sign_checks) != required:
                raise ArmBridgeError(
                    "All ± joint sign checks must pass first", code="sign_checks_incomplete"
                )
            if session.phase is not CommissioningPhase.READY:
                raise ArmBridgeError("Candidate requires a settled ready arm", code="invalid_state")
            measured = tuple(self._robot_state().arm_q[7:])
            session.candidate = {
                "profile_id": secrets.token_urlsafe(10),
                "label": self._required_text(label, "label"),
                "captured_at": _utc_now(),
                "measured_q": list(measured),
                "departed": False,
                "max_departure_rad": 0.0,
            }
            self._event("candidate_captured", session.candidate)
            return self.report()

    def replay_step(self, session_id: str, *, sequence: object) -> dict[str, Any]:
        """Move one joint by one jog toward the captured candidate."""

        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            if session.candidate is None:
                raise ArmBridgeError(
                    "No candidate pose has been captured", code="candidate_required"
                )
            if not session.candidate.get("departed"):
                raise ArmBridgeError(
                    "Move at least 0.02 rad away before replaying the candidate",
                    code="replay_departure_required",
                )
            measured = tuple(self._robot_state().arm_q[7:])
            candidate = tuple(float(value) for value in session.candidate["measured_q"])
            deltas = [target - actual for actual, target in zip(measured, candidate, strict=True)]
            index = max(range(7), key=lambda item: abs(deltas[item]))
            if abs(deltas[index]) <= self.config.settle_error_rad / 2.0:
                return self.report()
            return self.jog(
                session_id,
                sequence=sequence,
                joint_name=RIGHT_ARM_JOINT_NAMES[index],
                direction=1 if deltas[index] > 0 else -1,
                kind="replay",
            )

    def validate_replay(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._refresh_locked()
            if session.candidate is None or session.phase is not CommissioningPhase.READY:
                raise ArmBridgeError(
                    "No settled candidate is ready for validation", code="invalid_state"
                )
            if not session.candidate.get("departed"):
                raise ArmBridgeError(
                    "Replay must begin from a distinct measured pose",
                    code="replay_departure_required",
                )
            measured = tuple(self._robot_state().arm_q[7:])
            candidate = tuple(float(value) for value in session.candidate["measured_q"])
            if (
                max(
                    abs(actual - target) for actual, target in zip(measured, candidate, strict=True)
                )
                > self.config.settle_error_rad
            ):
                raise ArmBridgeError(
                    "Measured arm is not at the candidate pose", code="pose_not_reached"
                )
            session.replay_validations += 1
            session.candidate["departed"] = False
            session.candidate["max_departure_rad"] = 0.0
            self._event("replay_validated", {"count": session.replay_validations})
            return self.report()

    def promote(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            if session.candidate is None or session.replay_validations < 2:
                raise ArmBridgeError(
                    "Two successful replay validations are required", code="replay_required"
                )
            bridge_state = self.bridge.state_report()
            if bridge_state["state"] != ArmState.DISARMED.value:
                raise ArmBridgeError(
                    "Stop and fully disarm before promotion", code="arm_not_disarmed"
                )
            profile = {
                "schema_version": 1,
                "profile_id": session.candidate["profile_id"],
                "label": session.candidate["label"],
                "promoted_at": _utc_now(),
                "session_id": session.session_id,
                "joint_contract_id": joint_contract_id(),
                "joint_names": list(RIGHT_ARM_JOINT_NAMES),
                "measured_q": session.candidate["measured_q"],
                "robot": self.robot_identity,
                "sign_checks": session.sign_checks,
                "replay_validations": session.replay_validations,
            }
            encoded = json.dumps(profile, separators=(",", ":"), sort_keys=True).encode()
            profile["profile_hash"] = hashlib.sha256(encoded).hexdigest()
            _atomic_json(self.config.profile_path, profile)
            self._event(
                "profile_promoted",
                {
                    "profile_path": str(self.config.profile_path),
                    "profile_hash": profile["profile_hash"],
                },
            )
            return profile

    def stop(self, session_id: str, *, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self.bridge.stop(reason)
            session.phase = CommissioningPhase.STOPPED
            session.pending = None
            self._event("session_stopped", {"reason": reason})
            return self.report()

    def report(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            session = self.session
            if session is None:
                return {
                    "ok": True,
                    "active": False,
                    "joint_contract_id": joint_contract_id(),
                    "joint_contract": joint_contract(),
                    "bridge": self.bridge.state_report(),
                }
            baseline = session.baseline_q
            measured_state = self.bridge.hardware.latest_state()
            measured = None if measured_state is None else tuple(measured_state.arm_q[7:])
            bridge_report = self.bridge.state_report()
            commanded_arm = bridge_report.get("commanded_arm_q")
            pending = session.pending
            selected_joint = (
                None if pending is None else RIGHT_ARM_JOINT_NAMES[pending.joint_index]
            )
            visual = visualization_state(
                measured_body_q=None if measured_state is None else measured_state.body_q,
                measured_body_dq=None if measured_state is None else measured_state.body_dq,
                commanded_arm_q=commanded_arm if isinstance(commanded_arm, list) else None,
                reference_body_q=session.reference_body_q,
                received_at=None if measured_state is None else measured_state.received_at,
                now=self.monotonic(),
                state_ttl_s=self.bridge.config.state_ttl_s,
                selected_joint=selected_joint,
                faulted_joints=(
                    ()
                    if measured_state is None
                    else tuple(
                        name
                        for name in RIGHT_ARM_JOINT_NAMES
                        if any(name in fault for fault in measured_state.motor_faults)
                    )
                ),
            )
            return {
                "ok": session.phase is not CommissioningPhase.FAULT,
                "active": session.phase
                not in {CommissioningPhase.STOPPED, CommissioningPhase.FAULT},
                "session_id": session.session_id,
                "phase": session.phase.value,
                "joint_contract_id": joint_contract_id(),
                "joint_contract": joint_contract(),
                "movable_joint_names": list(self.config.movable_joint_names),
                "baseline_q": None if baseline is None else list(baseline),
                "stage_baseline_q": None
                if session.stage_baseline_q is None
                else list(session.stage_baseline_q),
                "measured_q": None if measured is None else list(measured),
                "session_delta_q": None
                if baseline is None or measured is None
                else [
                    round(actual - start, 6)
                    for actual, start in zip(measured, baseline, strict=True)
                ],
                "sign_checks": session.sign_checks,
                "checkpoints": session.checkpoints,
                "candidate": session.candidate,
                "replay_validations": session.replay_validations,
                "fault_reason": session.fault_reason,
                "bridge": bridge_report,
                "visualization": visual,
                "limits": {
                    "jog_step_rad": self.config.jog_step_rad,
                    "stage_limit_rad": self.config.stage_limit_rad,
                    "session_limit_rad_per_joint": self.config.session_limit_rad,
                },
            }

    def _refresh_locked(self) -> None:
        session = self.session
        if session is None:
            return
        bridge_report = self.bridge.state_report()
        bridge_state = str(bridge_report["state"])
        if bridge_state == ArmState.FAULT.value:
            if session.phase is not CommissioningPhase.FAULT:
                session.phase = CommissioningPhase.FAULT
                session.fault_reason = str(bridge_report.get("fault_reason") or "bridge_fault")
                self._event("session_fault", {"reason": session.fault_reason})
            return
        if session.phase is CommissioningPhase.ARMING and bridge_state == ArmState.ARMED.value:
            session.phase = CommissioningPhase.READY
            self._event("session_ready", {})
        pending = session.pending
        if session.phase is not CommissioningPhase.MOVING or pending is None:
            return
        now = self.monotonic()
        if now - pending.started_at > self.config.motion_timeout_s:
            self.bridge.stop("commissioning_motion_timeout")
            session.phase = CommissioningPhase.FAULT
            session.fault_reason = "motion_timeout"
            self._event("session_fault", {"reason": "motion_timeout"})
            return
        robot = self._robot_state()
        measured = tuple(robot.arm_q[7:])
        velocities = tuple(robot.arm_dq[7:])
        errors = [
            abs(actual - target) for actual, target in zip(measured, pending.target_q, strict=True)
        ]
        pending.peak_error_rad = max(pending.peak_error_rad, max(errors))
        nonselected_drift = max(
            (
                abs(measured[index] - pending.baseline_q[index])
                for index in range(7)
                if index != pending.joint_index
            ),
            default=0.0,
        )
        if nonselected_drift > self.config.nonselected_drift_rad:
            self.bridge.stop("commissioning_nonselected_drift")
            session.phase = CommissioningPhase.FAULT
            session.fault_reason = "nonselected_joint_drift"
            self._event("session_fault", {"reason": session.fault_reason})
            return
        settled = (
            max(errors) <= self.config.settle_error_rad
            and max(abs(value) for value in velocities) <= self.config.settle_velocity_rad_s
        )
        if settled:
            pending.settle_started_at = pending.settle_started_at or now
            if now - pending.settle_started_at >= self.config.settle_dwell_s:
                session.phase = CommissioningPhase.AWAITING_CONFIRM
                self._event("motion_settled", {"measured_q": list(measured), "errors_rad": errors})
        else:
            pending.settle_started_at = None

    def _event(self, event: str, details: dict[str, Any]) -> None:
        if self._event_path is None:
            return
        value = {"timestamp": _utc_now(), "event": event, **details}
        with self._event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        if self._summary_path is not None and self.session is not None:
            _atomic_json(
                self._summary_path,
                {
                    "schema_version": 1,
                    "session_id": self.session.session_id,
                    "updated_at": _utc_now(),
                    "phase": self.session.phase.value,
                    "event_counts": self._event_counts,
                    "sign_checks_passed": len(self.session.sign_checks),
                    "checkpoints": len(self.session.checkpoints),
                    "replay_validations": self.session.replay_validations,
                    "fault_reason": self.session.fault_reason,
                },
            )

    def _telemetry_loop(self) -> None:
        while not self._telemetry_stop.wait(0.05):
            with self._lock:
                self._refresh_locked()
                session = self.session
                path = self._telemetry_path
                if (
                    session is None
                    or path is None
                    or session.phase in {CommissioningPhase.STOPPED, CommissioningPhase.FAULT}
                ):
                    continue
                robot = self.bridge.hardware.latest_state()
                bridge = self.bridge.state_report()
                value = {
                    "timestamp": _utc_now(),
                    "session_id": session.session_id,
                    "phase": session.phase.value,
                    "measured_arm_q": None if robot is None else list(robot.arm_q),
                    "measured_arm_dq": None if robot is None else list(robot.arm_dq),
                    "commanded_arm_q": bridge.get("commanded_arm_q"),
                    "weight": bridge.get("weight"),
                    "standing": bridge.get("standing"),
                    "motion_mode_verified": bridge.get("motion_mode_verified"),
                    "controller_ownership_verified": bridge.get("controller_ownership_verified"),
                    "motor_state_healthy": bridge.get("motor_state_healthy"),
                    "loop": bridge.get("loop"),
                }
                try:
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
                        )
                except OSError:
                    # A daemon recorder must never affect the control path. The
                    # next event/summary still reports the operational state.
                    continue

    def _session(self, session_id: str) -> CommissioningSession:
        session = self.session
        if session is None or session.session_id != session_id:
            raise ArmBridgeError("Unknown commissioning session", code="session_mismatch")
        return session

    def _robot_state(self) -> Any:
        robot = self.bridge.hardware.latest_state()
        if robot is None:
            raise ArmBridgeError("Fresh LowState is required", code="robot_state_stale")
        return robot

    @staticmethod
    def _required_text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ArmBridgeError(f"{name} must be a non-empty string", code="invalid_request")
        return value.strip()

    @staticmethod
    def _sequence(session: CommissioningSession, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArmBridgeError("sequence must be an integer", code="invalid_sequence")
        if value <= session.last_client_sequence:
            raise ArmBridgeError("sequence must increase strictly", code="stale_sequence")
        return value
