"""Offline, movement-free helpers for inspecting and tuning G1 research runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from .arm_tracking.ik_solver import default_urdf_path
from .arm_tracking.joints import audit_urdf, joint_contract


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _session_paths(value: str | Path) -> tuple[Path, Path]:
    path = Path(value)
    if path.is_file() and path.name == "telemetry.jsonl":
        return path.parent, path
    if path.is_dir() and not (path / "telemetry.jsonl").is_file():
        sessions = sorted(
            (child for child in path.iterdir() if (child / "telemetry.jsonl").is_file()),
            key=lambda child: child.stat().st_mtime,
        )
        if sessions:
            path = sessions[-1]
    return path, path / "telemetry.jsonl"


def _expand_session_arguments(values: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in values:
        if path.is_dir() and not (path / "telemetry.jsonl").is_file():
            expanded.extend(
                sorted(child for child in path.iterdir() if (child / "telemetry.jsonl").is_file())
            )
        else:
            expanded.append(path)
    return expanded


def load_session(value: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory, telemetry_path = _session_paths(value)
    if not telemetry_path.is_file():
        raise ValueError(f"telemetry file not found: {telemetry_path}")
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {telemetry_path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"sample at {telemetry_path}:{line_number} is not an object")
        samples.append(value)
    if not samples:
        raise ValueError(f"telemetry file contains no samples: {telemetry_path}")
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    manifest["directory"] = str(directory)
    return manifest, samples


def _numeric(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]


def _tracking_numeric(samples: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for sample in samples:
        value = (sample.get("arm_tracking") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _metric(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.fmean(values), 3) if values else None,
        "p50": None if not values else round(_percentile(values, 50) or 0.0, 3),
        "p95": None if not values else round(_percentile(values, 95) or 0.0, 3),
        "max": None if not values else round(max(values), 3),
    }


def analyze_session(value: str | Path) -> dict[str, Any]:
    manifest, samples = load_session(value)
    reasons: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    track_ids: list[int] = []
    detection_samples = 0
    track_samples = 0
    for sample in samples:
        detection_samples += bool(sample.get("detections"))
        tracks = sample.get("tracks") or []
        track_samples += bool(tracks)
        if tracks and isinstance(tracks[0].get("track_id"), int):
            track_ids.append(int(tracks[0]["track_id"]))
        tracking = sample.get("arm_tracking") or {}
        status = str(tracking.get("status") or "unavailable")
        statuses[status] += 1
        if tracking.get("reason"):
            reasons[str(tracking["reason"])] += 1
    switches = sum(left != right for left, right in zip(track_ids, track_ids[1:]))
    count = len(samples)
    elapsed = max(float(sample.get("session_elapsed_s") or 0.0) for sample in samples)
    expected_fps = float(manifest.get("expected_fps") or 30.0)
    fps = _numeric(samples, "fps")
    yolo_fps = _numeric(samples, "yolo_fps")
    depth_age = _tracking_numeric(samples, "depth_age_ms")
    pair_skew = _tracking_numeric(samples, "pair_skew_ms")

    score_parts = {
        "camera_rate": min((statistics.fmean(fps) if fps else 0.0) / max(expected_fps, 1.0), 1.0),
        "inference_rate": min((statistics.fmean(yolo_fps) if yolo_fps else 0.0) / 10.0, 1.0),
        "rgb_depth_pairing": (
            0.0
            if not pair_skew
            else max(0.0, 1.0 - ((_percentile(pair_skew, 95) or 100.0) / 100.0))
        ),
        "depth_freshness": (
            0.0
            if not depth_age
            else max(0.0, 1.0 - ((_percentile(depth_age, 95) or 250.0) / 250.0))
        ),
        "track_availability": track_samples / count,
    }
    recommendations: list[dict[str, str]] = []
    suggested_experiments: list[str] = []

    def recommend(priority: str, area: str, finding: str, action: str) -> None:
        recommendations.append(
            {"priority": priority, "area": area, "finding": finding, "action": action}
        )

    mean_fps = statistics.fmean(fps) if fps else 0.0
    mean_yolo = statistics.fmean(yolo_fps) if yolo_fps else 0.0
    if mean_fps < expected_fps * 0.9:
        recommend(
            "high",
            "capture",
            f"mean camera rate {mean_fps:.1f} Hz is below 90% of {expected_fps:.1f} Hz",
            "Fix UDP/GStreamer loss or CPU contention before tuning tracking thresholds.",
        )
    if mean_yolo < 10.0:
        recommend(
            "medium",
            "inference",
            f"mean YOLO rate is {mean_yolo:.1f} Hz",
            "Run controlled IMGSZ=640, 800, 960 trials; select the smallest size that preserves labeled recall.",
        )
        suggested_experiments.extend(
            f"RESEARCH_LABEL=imgsz_{size} IMGSZ={size} ./scripts/run_gb10_vision_server.sh"
            for size in (640, 800, 960)
        )
    skew_p95 = _percentile(pair_skew, 95)
    if skew_p95 is not None and skew_p95 > 75.0:
        recommend(
            "high",
            "synchronization",
            f"RGB/depth skew p95 is {skew_p95:.1f} ms",
            "Reduce transport/queueing delay; do not raise the 100 ms safety gate to hide stale pairing.",
        )
    age_p95 = _percentile(depth_age, 95)
    if age_p95 is not None and age_p95 > 150.0:
        recommend(
            "high",
            "freshness",
            f"depth age p95 is {age_p95:.1f} ms",
            "Profile capture-to-IK stages and keep total target age below the 250 ms TTL.",
        )
    if track_samples / count < 0.7:
        recommend(
            "medium",
            "detection/tracking",
            f"tracks are present in only {track_samples / count:.1%} of samples",
            "Label misses first, then sweep CONF and tracker distance; do not select from unlabeled coverage alone.",
        )
        suggested_experiments.extend(
            f"RESEARCH_LABEL=conf_{int(confidence * 100):03d} CONF={confidence:.2f} "
            "./scripts/run_gb10_vision_server.sh"
            for confidence in (0.25, 0.35, 0.50)
        )
    if switches > max(2, len(track_ids) // 20):
        recommend(
            "medium",
            "tracking",
            f"the primary track changed ID {switches} times",
            "Replay the same scene while sweeping tracker match distance and missed-frame retention.",
        )
    for reason, occurrences in reasons.most_common(4):
        fraction = occurrences / count
        if fraction < 0.05:
            continue
        if reason.startswith("ik_") or reason == "ik_unavailable":
            action = "Repair and validate IK/URDF/joint-state seeding; this is not a threshold-tuning problem."
        elif reason == "depth_uncertain":
            action = "Record/replay RGB-Z16 failures and sweep ROI fraction, minimum support, MAD, and cluster gap."
        elif reason == "target_lost":
            action = "Label detector misses and ID switches, then sweep confidence and tracker retention."
        elif reason == "rgb_depth_pair_stale":
            action = "Fix stream synchronization and buffering; do not loosen the stale-pair gate."
        elif reason == "support_plane_clearance":
            action = "Inspect table segmentation and calibration before adjusting plane RANSAC or clearance."
        elif reason == "workspace_violation":
            action = (
                "Verify calibration and physically measured workspace bounds before changing them."
            )
        else:
            action = "Inspect representative samples before changing any gate."
        recommend(
            "high" if fraction >= 0.25 else "medium",
            "rejections",
            f"{reason} accounts for {fraction:.1%} of samples",
            action,
        )
    if not recommendations:
        recommend(
            "low",
            "next experiment",
            "no dominant telemetry bottleneck was detected",
            "Add ground-truth trial labels and compare controlled parameter sweeps before selecting a profile.",
        )

    return {
        "session": manifest.get("session_id") or Path(manifest["directory"]).name,
        "directory": manifest["directory"],
        "mode": manifest.get("mode"),
        "sample_count": count,
        "elapsed_s": round(elapsed, 3),
        "metrics": {
            "camera_fps": _metric(fps),
            "yolo_fps": _metric(yolo_fps),
            "depth_age_ms": _metric(depth_age),
            "pair_skew_ms": _metric(pair_skew),
            "detection_sample_fraction": round(detection_samples / count, 4),
            "track_sample_fraction": round(track_samples / count, 4),
            "primary_track_id_switches": switches,
            "statuses": dict(statuses),
            "rejection_reasons": dict(reasons),
        },
        "readiness_score": round(100.0 * statistics.fmean(score_parts.values()), 1),
        "score_notice": "Diagnostic observability score only; never authorizes robot movement.",
        "recommendations": recommendations,
        "suggested_experiments": list(dict.fromkeys(suggested_experiments)),
        "configuration": manifest.get("configuration", {}),
    }


def _print_analysis(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"Session: {report['session']} ({report.get('mode') or 'unknown mode'})")
    print(f"Samples: {report['sample_count']} over {report['elapsed_s']:.1f} s")
    print(f"Diagnostic score: {report['readiness_score']:.1f}/100 — not a movement authorization")
    for label, key, suffix in (
        ("Camera", "camera_fps", "Hz"),
        ("YOLO", "yolo_fps", "Hz"),
        ("Depth age", "depth_age_ms", "ms"),
        ("RGB/depth skew", "pair_skew_ms", "ms"),
    ):
        metric = metrics[key]
        print(f"{label:16} mean={metric['mean']} {suffix}  p95={metric['p95']} {suffix}")
    print(
        f"Detection coverage={metrics['detection_sample_fraction']:.1%}  "
        f"track coverage={metrics['track_sample_fraction']:.1%}  "
        f"ID switches={metrics['primary_track_id_switches']}"
    )
    print("Recommendations:")
    for item in report["recommendations"]:
        print(f"  [{item['priority'].upper()}] {item['area']}: {item['finding']}")
        print(f"         {item['action']}")
    if report["suggested_experiments"]:
        print("Copy/paste experiments (run the same labeled scene each time):")
        for command in report["suggested_experiments"]:
            print(f"  {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze G1 dry-run telemetry and audit the 29-DOF joint contract"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Explain one research session")
    analyze.add_argument("session", type=Path)
    analyze.add_argument("--json", action="store_true")
    compare = subparsers.add_parser("compare", help="Rank controlled dry-run sessions")
    compare.add_argument("sessions", nargs="+", type=Path)
    compare.add_argument("--json", action="store_true")
    joint = subparsers.add_parser("joint-audit", help="Verify SDK order and a G1 URDF")
    joint.add_argument("--urdf", type=Path)
    joint.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "analyze":
            report = analyze_session(args.session)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                _print_analysis(report)
            return 0
        if args.command == "compare":
            sessions = _expand_session_arguments(args.sessions)
            if len(sessions) < 2:
                raise ValueError("compare requires at least two research sessions")
            reports = sorted(
                (analyze_session(session) for session in sessions),
                key=lambda report: report["readiness_score"],
                reverse=True,
            )
            if args.json:
                print(json.dumps(reports, indent=2, sort_keys=True))
            else:
                print("Diagnostic ranking (requires identical labeled scenes):")
                for report in reports:
                    print(
                        f"  {report['readiness_score']:5.1f}  {report['session']}  "
                        f"tracks={report['metrics']['track_sample_fraction']:.1%}"
                    )
                print("This ranking does not validate accuracy or authorize movement.")
            return 0
        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = args.urdf or default_urdf_path(repo_root)
        report = audit_urdf(urdf_path)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Right-arm command order:")
            for item in joint_contract():
                print(
                    f"  q[{item['position']}] -> SDK[{item['sdk_index']}] "
                    f"{item['name']}  [{item['lower_rad']}, {item['upper_rad']}] rad"
                )
            print(f"URDF: {urdf_path}")
            print("PASS" if report["ok"] else "FAIL")
            for warning in report["warnings"]:
                print(f"  WARNING: {warning}")
            for error in report["errors"]:
                print(f"  ERROR: {error}")
        return 0 if report["ok"] else 2
    except ValueError as exc:
        print(f"g1-tune: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
