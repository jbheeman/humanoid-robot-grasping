#!/usr/bin/env python3
"""Produce a deterministic sim-to-real image audit and paired montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return np.asarray(image.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def stats(images: list[np.ndarray]) -> dict[str, object]:
    values = np.stack(images)
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    gx = np.diff(luminance, axis=2)
    gy = np.diff(luminance, axis=1)
    return {
        "rgb_mean": values.mean(axis=(0, 1, 2)).round(6).tolist(),
        "rgb_std": values.std(axis=(0, 1, 2)).round(6).tolist(),
        "luminance_mean": round(float(luminance.mean()), 6),
        "luminance_std": round(float(luminance.std()), 6),
        "edge_energy": round(float((np.abs(gx).mean() + np.abs(gy).mean()) / 2.0), 6),
    }


def euclidean_gap(left: object, right: object) -> float:
    return round(float(np.linalg.norm(np.asarray(left) - np.asarray(right))), 6)


def landmark_audit(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    errors: list[float] = []
    for pair in payload["pairs"]:
        real = np.asarray(pair["real_landmarks_xy"], dtype=float)
        synthetic = np.asarray(pair["synthetic_landmarks_xy"], dtype=float)
        if real.shape != synthetic.shape or real.ndim != 2 or real.shape[1] != 2:
            raise ValueError("landmark arrays must have matching [N,2] shapes")
        errors.extend(np.linalg.norm(real - synthetic, axis=1).tolist())
    return {
        "coordinate_system": payload.get("coordinate_system", "224px_preprocessed"),
        "landmark_count": len(errors),
        "mean_error": round(float(np.mean(errors)), 4),
        "p95_error": round(float(np.percentile(errors, 95)), 4),
    }


def verify_train_only(paths: list[Path], split_manifest: Path | None) -> None:
    if split_manifest is None:
        return
    splits = json.loads(split_manifest.read_text())
    forbidden = [str(path) for path in paths if splits.get(path.name) not in {None, "train"}]
    if forbidden:
        raise ValueError(f"validation/test images are forbidden during calibration: {forbidden[:3]}")


def montage(real_paths: list[Path], synthetic_paths: list[Path], output: Path) -> None:
    thumb = (448, 252)
    canvas = Image.new("RGB", (thumb[0] * 2, thumb[1] * len(real_paths) + 34), "#181818")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "REAL (train calibration only)", fill="white")
    draw.text((thumb[0] + 8, 8), "SYNTHETIC", fill="white")
    for row, (real, synthetic) in enumerate(zip(real_paths, synthetic_paths, strict=True)):
        y = 34 + row * thumb[1]
        for column, path in enumerate((real, synthetic)):
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = column * thumb[0] + (thumb[0] - image.width) // 2
            canvas.paste(image, (x, y + (thumb[1] - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--real-split-manifest", type=Path)
    parser.add_argument("--landmarks", type=Path)
    args = parser.parse_args()

    real = image_paths(args.real_dir)[: args.pairs]
    synthetic = image_paths(args.synthetic_dir)[: args.pairs]
    if len(real) != args.pairs or len(synthetic) != args.pairs:
        raise SystemExit(f"need {args.pairs} real and synthetic images; got {len(real)} and {len(synthetic)}")
    verify_train_only(real, args.real_split_manifest)
    real_stats = stats([preprocess(path) for path in real])
    synthetic_stats = stats([preprocess(path) for path in synthetic])
    report = {
        "pair_count": args.pairs,
        "preprocessing": "center_crop_224_lanczos_rgb",
        "real": real_stats,
        "synthetic": synthetic_stats,
        "gaps": {
            "rgb_mean_l2": euclidean_gap(real_stats["rgb_mean"], synthetic_stats["rgb_mean"]),
            "rgb_std_l2": euclidean_gap(real_stats["rgb_std"], synthetic_stats["rgb_std"]),
            "luminance_mean_abs": round(abs(real_stats["luminance_mean"] - synthetic_stats["luminance_mean"]), 6),
            "edge_energy_abs": round(abs(real_stats["edge_energy"] - synthetic_stats["edge_energy"]), 6),
        },
        "landmarks": landmark_audit(args.landmarks),
        "real_files": [str(path) for path in real],
        "synthetic_files": [str(path) for path in synthetic],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visual_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    montage(real, synthetic, args.output_dir / "paired_montage.jpg")
    print(json.dumps(report["gaps"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
