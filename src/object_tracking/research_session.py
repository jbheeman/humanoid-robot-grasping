from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class ResearchSummary:
    samples: int = 0
    target_sent: int = 0
    detection_observations: int = 0
    track_observations: int = 0
    fps_sum: float = 0.0
    yolo_fps_sum: float = 0.0
    depth_age_sum_ms: float = 0.0
    depth_age_samples: int = 0
    pair_skew_sum_ms: float = 0.0
    pair_skew_samples: int = 0
    max_depth_age_ms: float = 0.0
    max_pair_skew_ms: float = 0.0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def update(self, sample: dict[str, Any]) -> None:
        self.samples += 1
        self.fps_sum += float(sample.get("fps") or 0.0)
        self.yolo_fps_sum += float(sample.get("yolo_fps") or 0.0)
        self.detection_observations += len(sample.get("detections") or [])
        self.track_observations += len(sample.get("tracks") or [])
        tracking = sample.get("arm_tracking") or {}
        if tracking.get("status") == "target_sent":
            self.target_sent += 1
        if tracking.get("status") == "rejected" and tracking.get("reason"):
            self.rejection_reasons[str(tracking["reason"])] += 1
        depth_age = tracking.get("depth_age_ms")
        if isinstance(depth_age, (int, float)):
            self.depth_age_sum_ms += float(depth_age)
            self.depth_age_samples += 1
            self.max_depth_age_ms = max(self.max_depth_age_ms, float(depth_age))
        pair_skew = tracking.get("pair_skew_ms")
        if isinstance(pair_skew, (int, float)):
            self.pair_skew_sum_ms += float(pair_skew)
            self.pair_skew_samples += 1
            self.max_pair_skew_ms = max(self.max_pair_skew_ms, float(pair_skew))


class ResearchSession:
    """Bounded in-memory telemetry plus append-only JSONL research records."""

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: dict[str, Any],
        memory_samples: int = 500,
    ) -> None:
        started = datetime.now(timezone.utc)
        session_id = f"{started.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.session_id = session_id
        self.root = Path(root)
        self.directory = self.root / session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.telemetry_path = self.directory / "telemetry.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.manifest_path = self.directory / "manifest.json"
        self.started_at = started
        self.started_monotonic = time.monotonic()
        self.lock = threading.Lock()
        self.recent: deque[dict[str, Any]] = deque(maxlen=memory_samples)
        self.summary_state = ResearchSummary()
        self.last_summary_write = 0.0
        self.metadata = {
            "schema_version": 1,
            "session_id": session_id,
            "started_at": _utc_now(),
            "hostname": socket.gethostname(),
            "storage": {
                "telemetry": "telemetry.jsonl",
                "summary": "summary.json",
                "raw_images_recorded": False,
            },
            **metadata,
        }
        _write_json_atomic(self.manifest_path, self.metadata)
        self._write_summary_locked()

    def record(self, sample: dict[str, Any]) -> None:
        value = {
            "schema_version": 1,
            "sample_index": self.summary_state.samples,
            "recorded_at": _utc_now(),
            "session_elapsed_s": round(time.monotonic() - self.started_monotonic, 6),
            **sample,
        }
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        with self.lock:
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            self.recent.append(value)
            self.summary_state.update(value)
            now = time.monotonic()
            if now - self.last_summary_write >= 1.0:
                self._write_summary_locked()
                self.last_summary_write = now

    def session_report(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": True,
                "session_id": self.session_id,
                "started_at": self.metadata["started_at"],
                "elapsed_s": round(time.monotonic() - self.started_monotonic, 3),
                "directory": str(self.directory),
                "telemetry_file": str(self.telemetry_path),
                "summary_file": str(self.summary_path),
                "sample_count": self.summary_state.samples,
            }

    def summary_report(self) -> dict[str, Any]:
        with self.lock:
            return self._summary_locked()

    def recent_samples(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            count = max(1, min(int(limit), len(self.recent), 500))
            return list(self.recent)[-count:]

    def _summary_locked(self) -> dict[str, Any]:
        state = self.summary_state
        samples = max(state.samples, 1)
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "updated_at": _utc_now(),
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 3),
            "sample_count": state.samples,
            "target_sent_count": state.target_sent,
            "detection_observations": state.detection_observations,
            "track_observations": state.track_observations,
            "mean_camera_fps": round(state.fps_sum / samples, 3),
            "mean_yolo_fps": round(state.yolo_fps_sum / samples, 3),
            "mean_depth_age_ms": (
                None
                if state.depth_age_samples == 0
                else round(state.depth_age_sum_ms / state.depth_age_samples, 3)
            ),
            "max_depth_age_ms": round(state.max_depth_age_ms, 3),
            "mean_pair_skew_ms": (
                None
                if state.pair_skew_samples == 0
                else round(state.pair_skew_sum_ms / state.pair_skew_samples, 3)
            ),
            "max_pair_skew_ms": round(state.max_pair_skew_ms, 3),
            "rejection_reasons": dict(state.rejection_reasons),
        }

    def _write_summary_locked(self) -> None:
        _write_json_atomic(self.summary_path, self._summary_locked())
