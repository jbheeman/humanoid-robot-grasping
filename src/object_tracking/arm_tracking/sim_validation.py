"""Portable replay contract for simulation and physical-run diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1


def _finite_vector(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != length or not all(math.isfinite(item) for item in result):
        return None
    return result


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 3)


@dataclass(frozen=True)
class ReplayFrame:
    time_s: float
    status: str
    reason: str | None
    target_sequence: int | None
    right_arm_q_rad: tuple[float, ...] | None
    right_arm_tau_ff_nm: tuple[float, ...] | None
    measured_body_q_rad: tuple[float, ...] | None
    measured_body_dq_rad_s: tuple[float, ...] | None
    measured_right_arm_q_rad: tuple[float, ...] | None
    measured_right_arm_dq_rad_s: tuple[float, ...] | None
    object_xyz_m: tuple[float, ...] | None
    object_velocity_m_s: tuple[float, ...] | None
    target_xyz_m: tuple[float, ...] | None
    pipeline_age_ms: float | None
    ik_step_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 6),
            "status": self.status,
            "reason": self.reason,
            "target_sequence": self.target_sequence,
            "right_arm_q_rad": self.right_arm_q_rad,
            "right_arm_tau_ff_nm": self.right_arm_tau_ff_nm,
            "measured_body_q_rad": self.measured_body_q_rad,
            "measured_body_dq_rad_s": self.measured_body_dq_rad_s,
            "measured_right_arm_q_rad": self.measured_right_arm_q_rad,
            "measured_right_arm_dq_rad_s": self.measured_right_arm_dq_rad_s,
            "object_xyz_m": self.object_xyz_m,
            "object_velocity_m_s": self.object_velocity_m_s,
            "target_xyz_m": self.target_xyz_m,
            "pipeline_age_ms": self.pipeline_age_ms,
            "ik_step_type": self.ik_step_type,
        }


@dataclass(frozen=True)
class ReplayEpisode:
    source: str
    source_start_s: float
    source_end_s: float
    calibration_id: str | None
    support_plane: Mapping[str, Any] | None
    frames: tuple[ReplayFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "source_start_s": round(self.source_start_s, 6),
            "source_end_s": round(self.source_end_s, 6),
            "calibration_id": self.calibration_id,
            "coordinate_frame": "g1_torso_x_forward_y_left_z_up",
            "right_side_direction": [0.0, -1.0, 0.0],
            "support_plane": self.support_plane,
            "frames": [frame.to_dict() for frame in self.frames],
            "diagnostics": summarize_episode(self),
        }


def _arm_record(outer: Mapping[str, Any]) -> dict[str, Any]:
    arm = outer.get("arm_tracking")
    if not isinstance(arm, Mapping):
        return {}
    result = dict(arm)
    try:
        result["_elapsed_s"] = float(outer.get("session_elapsed_s"))
    except (TypeError, ValueError):
        return {}
    return result


def read_research_telemetry(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                outer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on telemetry line {line_number}") from exc
            if isinstance(outer, Mapping):
                arm = _arm_record(outer)
                if arm:
                    records.append(arm)
    if not records:
        raise ValueError("telemetry contains no arm_tracking records")
    return records


def episode_windows(
    records: Sequence[Mapping[str, Any]],
    *,
    split_gap_s: float = 3.0,
    tail_s: float = 2.0,
) -> tuple[tuple[float, float], ...]:
    """Find command episodes without depending on robot-console log parsing."""

    target_times = [
        float(record["_elapsed_s"]) for record in records if record.get("status") == "target_sent"
    ]
    if not target_times:
        return ()
    groups: list[list[float]] = [[target_times[0]]]
    for timestamp in target_times[1:]:
        if timestamp - groups[-1][-1] > split_gap_s:
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    earliest = float(records[0]["_elapsed_s"])
    latest = float(records[-1]["_elapsed_s"])
    return tuple(
        (
            max(earliest, group[0] - 1.0),
            min(latest, group[-1] + tail_s),
        )
        for group in groups
    )


def _latest_support_plane(records: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    latest = None
    for record in records:
        visual = record.get("visualization")
        if isinstance(visual, Mapping) and isinstance(visual.get("support_plane"), Mapping):
            latest = dict(visual["support_plane"])
    return latest


def build_replay_episode(
    source_path: str | Path,
    *,
    episode_index: int = -1,
    window: tuple[float, float] | None = None,
) -> ReplayEpisode:
    records = read_research_telemetry(source_path)
    if window is None:
        windows = episode_windows(records)
        if not windows:
            raise ValueError("telemetry contains no target_sent episode")
        try:
            window = windows[episode_index]
        except IndexError as exc:
            raise ValueError(
                f"episode index {episode_index} is outside {len(windows)} detected episodes"
            ) from exc
    start_s, end_s = window
    if not (math.isfinite(start_s) and math.isfinite(end_s) and start_s < end_s):
        raise ValueError("episode window must contain increasing finite times")
    selected = [record for record in records if start_s <= float(record["_elapsed_s"]) <= end_s]
    if not selected:
        raise ValueError("episode window contains no telemetry")
    frames: list[ReplayFrame] = []
    for record in selected:
        visual = record.get("visualization")
        measured = visual.get("measured_pose_rad") if isinstance(visual, Mapping) else None
        measured_velocity = (
            visual.get("measured_velocity_rad_s") if isinstance(visual, Mapping) else None
        )
        measured_body = _finite_vector(measured, 29)
        measured_body_velocity = _finite_vector(measured_velocity, 29)
        sequence = record.get("target_sequence")
        frames.append(
            ReplayFrame(
                time_s=float(record["_elapsed_s"]) - start_s,
                status=str(record.get("status") or "unknown"),
                reason=None if record.get("reason") is None else str(record["reason"]),
                target_sequence=(
                    int(sequence)
                    if isinstance(sequence, int) and not isinstance(sequence, bool)
                    else None
                ),
                right_arm_q_rad=_finite_vector(record.get("predicted_bounded_arm_command_rad"), 7),
                right_arm_tau_ff_nm=_finite_vector(record.get("gravity_feedforward_tau_nm"), 7),
                measured_body_q_rad=measured_body,
                measured_body_dq_rad_s=measured_body_velocity,
                measured_right_arm_q_rad=(None if measured_body is None else measured_body[22:29]),
                measured_right_arm_dq_rad_s=(
                    None if measured_body_velocity is None else measured_body_velocity[22:29]
                ),
                object_xyz_m=_finite_vector(record.get("object_xyz_m"), 3),
                object_velocity_m_s=_finite_vector(record.get("object_velocity_m_s"), 3),
                target_xyz_m=_finite_vector(record.get("target_xyz_m"), 3),
                pipeline_age_ms=(
                    float(record["pipeline_age_ms"])
                    if isinstance(record.get("pipeline_age_ms"), (int, float))
                    and math.isfinite(float(record["pipeline_age_ms"]))
                    else None
                ),
                ik_step_type=(
                    None if record.get("ik_step_type") is None else str(record["ik_step_type"])
                ),
            )
        )
    calibration_id = next(
        (str(record["calibration_id"]) for record in selected if record.get("calibration_id")),
        None,
    )
    return ReplayEpisode(
        source=str(Path(source_path)),
        source_start_s=start_s,
        source_end_s=end_s,
        calibration_id=calibration_id,
        support_plane=_latest_support_plane(selected),
        frames=tuple(frames),
    )


def summarize_episode(episode: ReplayEpisode) -> dict[str, Any]:
    sent = [
        frame
        for frame in episode.frames
        if frame.status == "target_sent" and frame.right_arm_q_rad is not None
    ]
    latencies = [frame.pipeline_age_ms for frame in sent if frame.pipeline_age_ms is not None]
    publish_rates = []
    for previous, current in zip(sent, sent[1:]):
        if (
            previous.target_sequence is not None
            and current.target_sequence is not None
            and current.time_s > previous.time_s
        ):
            publish_rates.append(
                (current.target_sequence - previous.target_sequence)
                / (current.time_s - previous.time_s)
            )
    right_margins = [
        frame.object_xyz_m[1] - frame.target_xyz_m[1]
        for frame in sent
        if frame.object_xyz_m is not None and frame.target_xyz_m is not None
    ]
    measured_speeds = [
        max(abs(value) for value in frame.measured_right_arm_dq_rad_s)
        for frame in episode.frames
        if frame.measured_right_arm_dq_rad_s is not None
    ]
    longest_gap_s = 0.0
    for previous, current in zip(sent, sent[1:]):
        longest_gap_s = max(longest_gap_s, current.time_s - previous.time_s)
    reasons: dict[str, int] = {}
    ik_step_times: dict[str, list[float]] = {}
    for frame in episode.frames:
        if frame.reason:
            reasons[frame.reason] = reasons.get(frame.reason, 0) + 1
        if frame.ik_step_type:
            ik_step_times.setdefault(frame.ik_step_type, []).append(frame.time_s)
    return {
        "frame_count": len(episode.frames),
        "target_frame_count": len(sent),
        "duration_s": round(episode.source_end_s - episode.source_start_s, 3),
        "estimated_target_publish_hz": (
            None if not publish_rates else round(median(publish_rates), 3)
        ),
        "longest_sampled_target_gap_s": round(longest_gap_s, 3),
        "pipeline_age_ms": {
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "p99": _quantile(latencies, 0.99),
        },
        "right_side_margin_m": {
            "median": None if not right_margins else round(median(right_margins), 5),
            "minimum": None if not right_margins else round(min(right_margins), 5),
            "fraction_on_right": (
                None
                if not right_margins
                else round(sum(value >= 0.0 for value in right_margins) / len(right_margins), 4)
            ),
        },
        "measured_max_joint_speed_rad_s": {
            "p50": _quantile(measured_speeds, 0.50),
            "p95": _quantile(measured_speeds, 0.95),
            "p99": _quantile(measured_speeds, 0.99),
            "maximum": None if not measured_speeds else round(max(measured_speeds), 3),
        },
        "rejection_reasons": reasons,
        "ik_step_duration_s": {
            name: round(max(times) - min(times), 3) for name, times in ik_step_times.items()
        },
        "support_plane_source": (
            None
            if episode.support_plane is None
            else (
                episode.support_plane.get("source")
                or (episode.support_plane.get("footprint") or {}).get("source")
            )
        ),
    }


def write_replay_episode(episode: ReplayEpisode, output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(episode.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
