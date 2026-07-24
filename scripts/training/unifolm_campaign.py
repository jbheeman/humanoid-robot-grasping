#!/usr/bin/env python3
"""Persistent UniFoLM train/evaluate/tune campaign controller.

The controller deliberately uses validation only for tuning.  A test result is
terminal and can never trigger another attempt.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable


SOURCES = ("g1_plush_touch_real", "g1_plush_touch_sim")
SOURCE_WEIGHTS = {"g1_plush_touch_real": 0.75, "g1_plush_touch_sim": 0.25}
TERMINAL_TEST_PHASES = {"complete", "test_rejected"}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value!r}")
    return number


def report_group(report: dict[str, Any], path: Path) -> str:
    representation = report.get("action_representation", "unknown")
    window = report.get("window_size", "unknown")
    stride = report.get("observation_stride", "unknown")
    if "v29-67real" in str(path):
        benchmark = "v29-67real-val96"
    else:
        benchmark = f"legacy-{path.parent.name}"
    return f"{benchmark}:{representation}:w{window}s{stride}"


def report_metrics(report: dict[str, Any]) -> tuple[float, float, dict[str, dict[str, Any]]]:
    normalized = 0.0
    raw = 0.0
    sources: dict[str, dict[str, Any]] = {}
    penalty = 0.0
    for source in SOURCES:
        metrics = report["sources"][source]
        model = metrics["model"]
        ade = finite(model["ade_m"])
        baseline = finite(metrics["best_baseline_ade_m"])
        if baseline <= 0:
            raise ValueError(f"{source}: invalid baseline {baseline}")
        weight = SOURCE_WEIGHTS[source]
        raw += weight * ade
        normalized += weight * ade / baseline
        saturation = finite(metrics["normalized_output_saturation_fraction"])
        magnitude = finite(metrics["predicted_to_target_displacement_ratio"])
        flags = metrics["diagnosis_flags"]
        penalty += weight * 2.0 * max(0.0, saturation - 0.05)
        penalty += weight * 0.25 * max(0.0, magnitude - 2.0)
        if not bool(flags["visually_conditioned"]):
            penalty += weight
        sources[source] = {
            "ade_m": ade,
            "fde_m": finite(model["fde_m"]),
            "median_m": finite(model["median_m"]),
            "p95_m": finite(model["p95_m"]),
            "baseline_ade_m": baseline,
            "normalized_ade": ade / baseline,
            "saturation_fraction": saturation,
            "magnitude_ratio": magnitude,
            "visually_conditioned": bool(flags["visually_conditioned"]),
            "action_active_ade_m": (
                finite(metrics["action_active_model"]["ade_m"])
                if metrics.get("action_active_model")
                else None
            ),
            "latency_error_m": (
                metrics.get("first_post_latency_waypoint", {}).get("mean_error_m")
            ),
        }
    return normalized + penalty, raw, sources


class CampaignDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                run_id TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                seed INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                warmup_steps INTEGER NOT NULL,
                real_weight REAL NOT NULL,
                image_aug INTEGER NOT NULL,
                init_mode TEXT NOT NULL,
                parent_checkpoint TEXT,
                max_steps INTEGER NOT NULL,
                observed_unix REAL,
                started_unix REAL,
                ended_unix REAL,
                objective REAL,
                weighted_ade_m REAL,
                comparable_group TEXT,
                selected_checkpoint TEXT,
                pipeline_status TEXT,
                detail TEXT,
                infrastructure_failure INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS source_metrics (
                attempt_id TEXT NOT NULL,
                source TEXT NOT NULL,
                ade_m REAL,
                fde_m REAL,
                median_m REAL,
                p95_m REAL,
                baseline_ade_m REAL,
                normalized_ade REAL,
                saturation_fraction REAL,
                magnitude_ratio REAL,
                visually_conditioned INTEGER,
                action_active_ade_m REAL,
                latency_error_m REAL,
                PRIMARY KEY (attempt_id, source)
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_unix REAL NOT NULL,
                kind TEXT NOT NULL,
                attempt_id TEXT,
                payload_json TEXT NOT NULL
            );
            """
        )
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(attempts)")
        }
        if "observed_unix" not in columns:
            self.connection.execute("ALTER TABLE attempts ADD COLUMN observed_unix REAL")
        if "resume_checkpoint" not in columns:
            self.connection.execute(
                "ALTER TABLE attempts ADD COLUMN resume_checkpoint TEXT"
            )
        self.connection.commit()

    def set_meta(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, sort_keys=True)
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, encoded),
        )
        self.connection.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return default if row is None else json.loads(row["value"])

    def event(self, kind: str, attempt_id: str | None, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO events(timestamp_unix,kind,attempt_id,payload_json) "
            "VALUES(?,?,?,?)",
            (time.time(), kind, attempt_id, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def attempts(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute("SELECT * FROM attempts ORDER BY ordinal")
        )

    def comparable_attempts(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM attempts WHERE objective IS NOT NULL "
                "AND infrastructure_failure=0 ORDER BY ordinal"
            )
        )

    def upsert_attempt(self, values: dict[str, Any]) -> None:
        columns = list(values)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns[1:])
        self.connection.execute(
            f"INSERT INTO attempts({','.join(columns)}) "
            f"VALUES({','.join('?' for _ in columns)}) "
            f"ON CONFLICT(attempt_id) DO UPDATE SET {updates}",
            [values[column] for column in columns],
        )
        self.connection.commit()

    def update_attempt(self, attempt_id: str, **updates: Any) -> None:
        assignments = ", ".join(f"{key}=?" for key in updates)
        self.connection.execute(
            f"UPDATE attempts SET {assignments} WHERE attempt_id=?",
            [*updates.values(), attempt_id],
        )
        self.connection.commit()

    def record_report(
        self,
        attempt_id: str,
        report_path: Path,
        pipeline_status: Path | None = None,
    ) -> tuple[float, float]:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        objective, raw, sources = report_metrics(report)
        checkpoint = str(Path(report["checkpoint"]).resolve())
        self.update_attempt(
            attempt_id,
            objective=objective,
            weighted_ade_m=raw,
            comparable_group=report_group(report, report_path),
            selected_checkpoint=checkpoint,
            pipeline_status=str(pipeline_status) if pipeline_status else None,
        )
        for source, metrics in sources.items():
            columns = ["attempt_id", "source", *metrics]
            values = [attempt_id, source, *metrics.values()]
            self.connection.execute(
                f"INSERT OR REPLACE INTO source_metrics({','.join(columns)}) "
                f"VALUES({','.join('?' for _ in columns)})",
                values,
            )
        self.connection.commit()
        self.event(
            "metrics_recorded",
            attempt_id,
            {"report": str(report_path), "objective": objective, "weighted_ade_m": raw},
        )
        return objective, raw


def attempt_id(campaign: str, ordinal: int) -> str:
    return f"{campaign}:{ordinal:04d}"


def next_parameters(
    db: CampaignDB,
    campaign: str,
    base_checkpoint: Path,
    max_steps: int,
) -> dict[str, Any]:
    rows = [
        row
        for row in db.comparable_attempts()
        if row["selected_checkpoint"]
        and Path(row["selected_checkpoint"]).is_file()
    ]
    ordinal = max((row["ordinal"] for row in db.attempts()), default=0) + 1
    if not rows:
        return {
            "attempt_id": attempt_id(campaign, ordinal),
            "ordinal": ordinal,
            "run_id": f"{campaign}-attempt{ordinal}",
            "state": "PLANNED",
            "seed": 42 + ordinal - 1,
            "learning_rate": 2.0e-5,
            "warmup_steps": 200,
            "real_weight": 3.0,
            "image_aug": 1,
            "init_mode": "action_warm_start",
            "parent_checkpoint": str(base_checkpoint),
            "max_steps": max_steps,
            "observed_unix": time.time(),
        }
    best = min(rows, key=lambda row: finite(row["objective"]))
    ordinal = max(row["ordinal"] for row in db.attempts()) + 1
    cycle = (ordinal - 1) % 6
    learning_rates = (1.0e-5, 5.0e-6, 2.5e-6)
    learning_rate = learning_rates[min(cycle // 2, 2)]
    real_weight = min(9.0, max(2.0, float(best["real_weight"])))
    source_rows = list(
        db.connection.execute(
            "SELECT * FROM source_metrics WHERE attempt_id=?", (best["attempt_id"],)
        )
    )
    by_source = {row["source"]: row for row in source_rows}
    if by_source:
        real = finite(by_source["g1_plush_touch_real"]["normalized_ade"])
        sim = finite(by_source["g1_plush_touch_sim"]["normalized_ade"])
        if real > 1.1 * sim:
            real_weight = min(9.0, real_weight + 1.0)
        elif sim > 1.1 * real:
            real_weight = max(2.0, real_weight - 0.5)
    fresh = cycle in (2, 5)
    return {
        "attempt_id": attempt_id(campaign, ordinal),
        "ordinal": ordinal,
        "run_id": f"{campaign}-attempt{ordinal}",
        "state": "PLANNED",
        "seed": 42 + ordinal - 1,
        "learning_rate": learning_rate,
        "warmup_steps": (200, 400, 400, 800, 400, 800)[cycle],
        "real_weight": real_weight,
        "image_aug": 0 if cycle == 4 else 1,
        "init_mode": "fresh" if fresh else "action_warm_start",
        "parent_checkpoint": None if fresh else str(best["selected_checkpoint"]),
        "max_steps": max_steps,
        "observed_unix": time.time(),
    }


def process_identity(process: subprocess.Popen[Any]) -> dict[str, Any]:
    return {"pid": process.pid, "started_unix": time.time()}


def run_process(
    command: list[str],
    *,
    environment: dict[str, str],
    campaign_dir: Path,
    run_dir: Path,
    status_path: Path,
) -> tuple[int, str | None]:
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
    )
    atomic_json(status_path, {"phase": "running", **process_identity(process)})
    stop_reason = None
    last_state_scan = 0.0
    while process.poll() is None:
        if (campaign_dir / "STOP_NOW").exists():
            stop_reason = "stop-now"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            break
        if time.time() - last_state_scan >= 30.0:
            verify_training_states(run_dir, minimum_age_seconds=30.0)
            last_state_scan = time.time()
        time.sleep(2)
    return int(process.wait()), stop_reason


def state_step(path: Path) -> int:
    return int(path.name.split("_")[1])


def verify_training_states(
    run_dir: Path,
    *,
    minimum_age_seconds: float = 0.0,
    create_manifests: bool = True,
) -> list[Path]:
    verified = []
    for path in sorted(
        run_dir.glob("checkpoints/steps_*_training_state"), key=state_step
    ):
        manifest = path / "COMPLETE.json"
        if manifest.is_file():
            verified.append(path)
            continue
        if not create_manifests:
            continue
        model_files = list(path.rglob("*model_states.pt"))
        optimizer_files = list(path.rglob("*optim_states.pt"))
        files = model_files + optimizer_files
        if (
            not model_files
            or not optimizer_files
            or any(item.stat().st_size < 1024 * 1024 for item in files)
            or time.time() - max(item.stat().st_mtime for item in files)
            < minimum_age_seconds
        ):
            continue
        sizes = [item.stat().st_size for item in files]
        time.sleep(0.1)
        if sizes != [item.stat().st_size for item in files]:
            continue
        payload = {
            "schema_version": 1,
            "step": state_step(path),
            "files": {
                str(item.relative_to(path)): item.stat().st_size for item in files
            },
            "verified_unix": time.time(),
        }
        atomic_json(manifest, payload)
        verified.append(path)
    return verified


def quarantine(path: Path, trash: Path) -> None:
    if not path.exists():
        return
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / f"{path.parent.name}--{path.name}"
    if target.exists():
        target = trash / f"{target.name}--{int(time.time())}"
    os.replace(path, target)


def apply_retention(
    db: CampaignDB,
    workspace: Path,
    campaign: str,
    *,
    trash_grace_hours: float,
) -> None:
    ranked = sorted(
        db.comparable_attempts(), key=lambda row: finite(row["objective"])
    )
    rank = {row["attempt_id"]: index + 1 for index, row in enumerate(ranked)}
    trash = workspace / "runs/.trash" / campaign
    for row in db.attempts():
        if row["state"] not in ("COMPLETE", "VALIDATION_REJECTED", "TEST_REJECTED"):
            continue
        run_dir = workspace / "runs/unifolm_plush_touch" / row["run_id"]
        checkpoints = run_dir / "checkpoints"
        if not checkpoints.is_dir():
            continue
        selected = Path(row["selected_checkpoint"]).resolve() if row["selected_checkpoint"] else None
        position = rank.get(row["attempt_id"], 10**9)
        action_paths = sorted(checkpoints.glob("steps_*_action_model.pt"))
        states = verify_training_states(run_dir)
        keep_actions = set(action_paths if position == 1 else ([selected] if selected else []))
        keep_state_count = 2 if position == 1 else (1 if position <= 3 else 0)
        keep_states = set(states[-keep_state_count:] if keep_state_count else [])
        for path in action_paths:
            if path.resolve() not in keep_actions:
                quarantine(path, trash)
        for path in states:
            if path not in keep_states:
                quarantine(path, trash)
    cutoff = time.time() - trash_grace_hours * 3600.0
    if trash.is_dir():
        for path in trash.iterdir():
            if path.stat().st_mtime < cutoff:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def export_metrics(db: CampaignDB, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(
        db.connection.execute(
            "SELECT a.*,m.source,m.ade_m,m.fde_m,m.median_m,m.p95_m,"
            "m.baseline_ade_m,m.normalized_ade,m.saturation_fraction,"
            "m.magnitude_ratio,m.visually_conditioned,m.action_active_ade_m,"
            "m.latency_error_m FROM attempts a LEFT JOIN source_metrics m "
            "ON a.attempt_id=m.attempt_id ORDER BY a.ordinal,m.source"
        )
    )
    dictionaries = [dict(row) for row in rows]
    jsonl = output_dir / "metrics.jsonl"
    temporary = jsonl.with_suffix(".jsonl.partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dictionaries),
        encoding="utf-8",
    )
    os.replace(temporary, jsonl)
    csv_path = output_dir / "metrics.csv"
    csv_temporary = csv_path.with_suffix(".csv.partial")
    fieldnames = list(dictionaries[0]) if dictionaries else ["attempt_id"]
    with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dictionaries)
    os.replace(csv_temporary, csv_path)


def plot_metrics(db: CampaignDB, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = db.comparable_attempts()
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(row["comparable_group"] or "unknown", []).append(row)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for group, values in groups.items():
        values.sort(key=lambda row: row["observed_unix"] or row["ordinal"])
        x = [row["ordinal"] for row in values]
        raw = [1000.0 * finite(row["weighted_ade_m"]) for row in values]
        objective = [finite(row["objective"]) for row in values]
        axes[0].plot(x, raw, marker="o", label=group)
        axes[1].plot(x, objective, marker="o", label=group)
        axes[1].plot(
            x,
            [min(objective[: index + 1]) for index in range(len(objective))],
            linestyle="--",
            alpha=0.7,
            label=f"{group} best",
        )
    axes[0].set_ylabel("75/25 weighted ADE (mm)")
    axes[1].set_ylabel("baseline-normalized objective")
    axes[1].set_xlabel("attempt ordinal")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"campaign_progress.{extension}", dpi=180)
    plt.close(figure)


def infer_run_id(report: dict[str, Any], path: Path) -> str:
    checkpoint = report.get("checkpoint")
    if checkpoint:
        candidate = Path(checkpoint)
        parts = candidate.parts
        if "unifolm_plush_touch" in parts:
            index = parts.index("unifolm_plush_touch")
            if index + 1 < len(parts):
                return parts[index + 1]
    return path.stem


def backfill(db: CampaignDB, root: Path, campaign: str) -> int:
    reports = sorted(root.glob("**/*_val.json"))
    candidates: dict[str, list[tuple[int, float, Path, dict[str, Any]]]] = {}
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not all(source in report.get("sources", {}) for source in SOURCES):
                continue
            run_id = infer_run_id(report, report_path)
            name = report_path.name
            priority = (
                3
                if "strong_val" in name
                else 2
                if "selected" in name
                else 1
                if "fast_val" not in name
                else 0
            )
            candidates.setdefault(run_id, []).append(
                (priority, report_path.stat().st_mtime, report_path, report)
            )
        except (OSError, json.JSONDecodeError):
            continue
    imported = 0
    known = {row["run_id"] for row in db.attempts()}
    ordinal = max([row["ordinal"] for row in db.attempts()] or [0])
    selected_reports = [
        max(values, key=lambda item: (item[0], item[1]))
        for values in candidates.values()
    ]
    selected_reports.sort(key=lambda item: item[1])
    for _, observed_unix, report_path, report in selected_reports:
        try:
            run_id = infer_run_id(report, report_path)
            if run_id in known:
                continue
            ordinal += 1
            identifier = f"historical:{hashlib.sha1(str(report_path).encode()).hexdigest()[:12]}"
            db.upsert_attempt(
                {
                    "attempt_id": identifier,
                    "ordinal": ordinal,
                    "run_id": run_id,
                    "state": "IMPORTED",
                    "seed": int(report.get("seed", 0)),
                    "learning_rate": 0.0,
                    "warmup_steps": 0,
                    "real_weight": 3.0,
                    "image_aug": 0,
                    "init_mode": "historical_import",
                    "parent_checkpoint": None,
                    "max_steps": 0,
                    "observed_unix": observed_unix,
                    "detail": f"imported:{report_path}",
                }
            )
            db.record_report(identifier, report_path)
            known.add(run_id)
            imported += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
    db.event("historical_backfill", None, {"reports_imported": imported})
    return imported


def write_campaign_status(db: CampaignDB, campaign_dir: Path, phase: str) -> None:
    attempts = db.attempts()
    latest = dict(attempts[-1]) if attempts else None
    atomic_json(
        campaign_dir / "CAMPAIGN_STATUS.json",
        {
            "schema_version": 1,
            "phase": phase,
            "latest_attempt": latest,
            "attempt_count": len(attempts),
            "updated_unix": time.time(),
            "robot_execution_authorized": False,
        },
    )


def run_campaign(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    campaign_dir = workspace / "runs/automation" / args.campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (campaign_dir / "controller.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another campaign controller is already running")
    db = CampaignDB(campaign_dir / "campaign.sqlite3")
    db.set_meta("campaign", args.campaign)
    db.set_meta("test_terminal", db.get_meta("test_terminal", False))
    if db.get_meta("test_terminal", False):
        write_campaign_status(db, campaign_dir, "TEST_TERMINAL")
        return 0
    stop_now = campaign_dir / "STOP_NOW"
    stop_after = campaign_dir / "STOP_AFTER_ATTEMPT"
    if stop_now.exists():
        write_campaign_status(db, campaign_dir, "PAUSED")
        return 0
    latest_rows = db.attempts()
    if latest_rows and latest_rows[-1]["state"] in ("TRAINING", "VALIDATING"):
        db.update_attempt(
            latest_rows[-1]["attempt_id"],
            state="INFRASTRUCTURE_FAILED",
            ended_unix=time.time(),
            detail="controller restarted after an unclean attempt exit",
            infrastructure_failure=1,
        )
        write_campaign_status(db, campaign_dir, "INFRASTRUCTURE_PAUSED")
        return 32

    comparable = db.comparable_attempts()
    best_objective = min(
        (finite(row["objective"]) for row in comparable), default=math.inf
    )
    stale = int(db.get_meta("non_improving_attempts", 0))
    campaign_attempts = sum(
        row["attempt_id"].startswith(f"{args.campaign}:")
        for row in db.attempts()
    )
    while stale < args.patience and campaign_attempts < args.max_attempts:
        stopped_rows = [
            row for row in db.attempts() if row["state"] == "STOPPED"
        ]
        if stopped_rows:
            stopped_row = stopped_rows[-1]
            resume_checkpoint = stopped_row["resume_checkpoint"]
            if not resume_checkpoint or not Path(resume_checkpoint).is_dir():
                write_campaign_status(db, campaign_dir, "PAUSED_NO_RESUME_STATE")
                return 31
            parameters = dict(stopped_row)
            parameters["state"] = "PLANNED"
            parameters["ended_unix"] = None
            db.update_attempt(
                parameters["attempt_id"], state="PLANNED", ended_unix=None
            )
            is_resume = True
        else:
            parameters = next_parameters(
                db, args.campaign, args.initial_checkpoint.resolve(), args.max_steps
            )
            db.upsert_attempt(parameters)
            is_resume = False
            campaign_attempts += 1
        identifier = parameters["attempt_id"]
        run_id = parameters["run_id"]
        run_dir = workspace / "runs/unifolm_plush_touch" / run_id
        db.update_attempt(identifier, state="TRAINING", started_unix=time.time())
        db.event("attempt_started", identifier, parameters)
        write_campaign_status(db, campaign_dir, "TRAINING")

        environment = dict(os.environ)
        environment.update(
            {
                "RUN_ID": run_id,
                "MAX_TRAIN_STEPS": str(parameters["max_steps"]),
                "TRAINING_SEED": str(parameters["seed"]),
                "TRAINING_GPUS": args.training_gpus,
                "ACTION_MODEL_LR": str(parameters["learning_rate"]),
                "WARMUP_STEPS": str(parameters["warmup_steps"]),
                "REAL_SAMPLE_WEIGHT": str(parameters["real_weight"]),
                "IMAGE_AUG": "true" if parameters["image_aug"] else "false",
                "ACTION_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
                "FULL_STATE_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
                "FULL_STATE_KEEP_LAST": "2",
                "INITIAL_ACTION_CHECKPOINT": (
                    parameters["parent_checkpoint"]
                    if not is_resume
                    and parameters["init_mode"] == "action_warm_start"
                    else ""
                ),
                "RESUME_FROM_CHECKPOINT": (
                    parameters["resume_checkpoint"] if is_resume else ""
                ),
            }
        )
        returncode, stopped = run_process(
            [str(workspace / "scripts/training/run_unifolm_v29_67real.sh"), "full"],
            environment=environment,
            campaign_dir=campaign_dir,
            run_dir=run_dir,
            status_path=campaign_dir / "PROCESS_STATUS.json",
        )
        verified_states = verify_training_states(
            run_dir, create_manifests=stopped is None
        )
        if stopped:
            db.update_attempt(
                identifier,
                state="STOPPED",
                ended_unix=time.time(),
                detail=(
                    f"stopped; latest verified state: {verified_states[-1]}"
                    if verified_states
                    else "stopped; no verified full state"
                ),
                resume_checkpoint=(
                    str(verified_states[-1]) if verified_states else None
                ),
            )
            write_campaign_status(db, campaign_dir, "PAUSED")
            export_metrics(db, campaign_dir / "exports")
            return 0
        if returncode != 0:
            db.update_attempt(
                identifier,
                state="INFRASTRUCTURE_FAILED",
                ended_unix=time.time(),
                detail=f"training exit {returncode}",
                infrastructure_failure=1,
            )
            write_campaign_status(db, campaign_dir, "INFRASTRUCTURE_PAUSED")
            export_metrics(db, campaign_dir / "exports")
            return returncode

        db.update_attempt(identifier, state="VALIDATING")
        write_campaign_status(db, campaign_dir, "VALIDATING")
        gpu_args = args.training_gpus.split(",")
        evaluation = subprocess.run(
            [
                str(args.python),
                str(workspace / "scripts/training/evaluate_unifolm_run.py"),
                "--root",
                str(workspace),
                "--run-id",
                run_id,
                "--config",
                str(args.config),
                "--data-root",
                str(args.data_root),
                "--statistics",
                str(args.statistics),
                "--expected-final-step",
                str(parameters["max_steps"]),
                "--samples-per-source",
                str(args.samples),
                "--real-weight",
                "0.75",
                "--gpus",
                *gpu_args,
            ],
            check=False,
        )
        pipeline_status = workspace / "runs/diagnostics" / run_id / "PIPELINE_STATUS.json"
        if not pipeline_status.is_file():
            db.update_attempt(
                identifier,
                state="INFRASTRUCTURE_FAILED",
                ended_unix=time.time(),
                detail=f"evaluation exit {evaluation.returncode}; no status",
                infrastructure_failure=1,
            )
            write_campaign_status(db, campaign_dir, "INFRASTRUCTURE_PAUSED")
            return 30
        status = json.loads(pipeline_status.read_text(encoding="utf-8"))
        phase = status["phase"]
        selected = status.get("selected")
        if selected:
            strong = Path(status["strong_validation_report"])
            objective, _ = db.record_report(identifier, strong, pipeline_status)
        else:
            objective = math.inf
        state = {
            "validation_rejected": "VALIDATION_REJECTED",
            "complete": "COMPLETE",
            "test_rejected": "TEST_REJECTED",
        }.get(phase, "INFRASTRUCTURE_FAILED")
        db.update_attempt(identifier, state=state, ended_unix=time.time(), detail=phase)
        if phase in TERMINAL_TEST_PHASES:
            db.set_meta("test_terminal", True)
            db.event("test_terminal", identifier, {"phase": phase})
            apply_retention(
                db,
                workspace,
                args.campaign,
                trash_grace_hours=args.trash_grace_hours,
            )
            export_metrics(db, campaign_dir / "exports")
            plot_metrics(db, campaign_dir / "graphs")
            write_campaign_status(db, campaign_dir, "TEST_TERMINAL")
            return 0 if phase == "complete" else 2

        threshold = (
            args.min_absolute_improvement
            if not math.isfinite(best_objective)
            else max(
                args.min_absolute_improvement,
                args.min_relative_improvement * abs(best_objective),
            )
        )
        if math.isfinite(objective) and (
            not math.isfinite(best_objective)
            or best_objective - objective >= threshold
        ):
            best_objective = objective
            stale = 0
            db.event("objective_improved", identifier, {"objective": objective})
        else:
            stale += 1
        db.set_meta("non_improving_attempts", stale)
        apply_retention(
            db,
            workspace,
            args.campaign,
            trash_grace_hours=args.trash_grace_hours,
        )
        export_metrics(db, campaign_dir / "exports")
        plot_metrics(db, campaign_dir / "graphs")
        if stop_after.exists():
            write_campaign_status(db, campaign_dir, "PAUSED_AFTER_ATTEMPT")
            return 0
    phase = (
        "ATTEMPT_LIMIT_PAUSED"
        if campaign_attempts >= args.max_attempts
        else "NO_PROGRESS_PAUSED"
    )
    write_campaign_status(db, campaign_dir, phase)
    return 0


def control(args: argparse.Namespace) -> int:
    campaign_dir = args.workspace.resolve() / "runs/automation" / args.campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "resume":
        for name in ("STOP_NOW", "STOP_AFTER_ATTEMPT"):
            (campaign_dir / name).unlink(missing_ok=True)
        print(f"cleared stop markers in {campaign_dir}")
        return 0
    marker = campaign_dir / (
        "STOP_NOW" if args.command == "stop-now" else "STOP_AFTER_ATTEMPT"
    )
    marker.write_text(f"{time.time()}\n", encoding="utf-8")
    print(marker)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/home/aarav/Documents/g1-bunny-vla-workspace"),
    )
    parser.add_argument("--campaign", default="v29-67real-adaptive")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--initial-checkpoint", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--statistics", type=Path, required=True)
    run.add_argument(
        "--python",
        type=Path,
        default=Path("/home/aarav/miniconda3/envs/g1-unifolm-train/bin/python"),
    )
    run.add_argument("--training-gpus", default="1")
    run.add_argument("--max-steps", type=int, default=4000)
    run.add_argument("--checkpoint-interval", type=int, default=1000)
    run.add_argument("--samples", type=int, default=96)
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--max-attempts", type=int, default=24)
    run.add_argument("--min-relative-improvement", type=float, default=0.01)
    run.add_argument("--min-absolute-improvement", type=float, default=0.01)
    run.add_argument("--trash-grace-hours", type=float, default=24.0)
    subparsers.add_parser("stop-now")
    subparsers.add_parser("stop-after-attempt")
    subparsers.add_parser("resume")
    subparsers.add_parser("status")
    subparsers.add_parser("backfill")
    subparsers.add_parser("export")
    subparsers.add_parser("plot")
    args = parser.parse_args()
    if args.command == "run":
        return run_campaign(args)
    if args.command in ("stop-now", "stop-after-attempt", "resume"):
        return control(args)
    campaign_dir = args.workspace.resolve() / "runs/automation" / args.campaign
    db = CampaignDB(campaign_dir / "campaign.sqlite3")
    if args.command == "status":
        status = campaign_dir / "CAMPAIGN_STATUS.json"
        print(status.read_text() if status.is_file() else "{}")
    elif args.command == "backfill":
        count = backfill(
            db, args.workspace.resolve() / "runs/diagnostics", args.campaign
        )
        export_metrics(db, campaign_dir / "exports")
        plot_metrics(db, campaign_dir / "graphs")
        print(f"imported={count}")
    elif args.command == "export":
        export_metrics(db, campaign_dir / "exports")
    elif args.command == "plot":
        plot_metrics(db, campaign_dir / "graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
