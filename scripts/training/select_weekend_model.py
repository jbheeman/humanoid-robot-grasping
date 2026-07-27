#!/usr/bin/env python3
"""Select the best weekend checkpoint and export its 960px TensorRT engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from ultralytics import YOLO


def evaluate(model_path: Path, data: Path, run_dir: Path) -> dict[str, float]:
    metrics = YOLO(model_path).val(
        data=str(data), imgsz=1280, batch=8, device=0, workers=14, verbose=False,
        project=str(run_dir), name=model_path.parent.parent.name, exist_ok=True,
    )
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--polish", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [path for path in (args.primary, args.polish) if path and path.is_file()]
    if not candidates:
        raise SystemExit("No candidate checkpoint exists")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores = {str(path): evaluate(path, args.data.resolve(), output / "validation") for path in candidates}
    winner = max(candidates, key=lambda path: scores[str(path)]["map50_95"])
    weights = output / "weights"
    weights.mkdir(exist_ok=True)
    selected = weights / "best.pt"
    shutil.copy2(winner, selected)
    summary: dict[str, object] = {"winner": str(winner), "scores": scores, "selected": str(selected)}
    try:
        engine = YOLO(selected).export(format="engine", imgsz=960, half=True, dynamic=False, device=0)
        summary["engine_960_fp16"] = str(engine)
    except Exception as exc:  # Keep the selected checkpoint usable even if TensorRT is unavailable.
        summary["engine_export_error"] = f"{type(exc).__name__}: {exc}"
    (output / "weekend_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
