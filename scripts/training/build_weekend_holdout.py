#!/usr/bin/env python3
"""Create a reproducible, source-grouped validation holdout for weekend training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AUGMENT_PREFIXES = (
    "aug_blur_jpeg_",
    "aug_motion_noise_",
    "aug_g1_gray_720p_",
)


def canonical_name(path: Path) -> str:
    stem = path.stem
    for prefix in AUGMENT_PREFIXES:
        if stem.startswith(prefix):
            return re.sub(r"_\d{2}$", "", stem[len(prefix) :])
    return stem


def label_path(image: Path) -> Path:
    return image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"


def has_label(image: Path) -> bool:
    path = label_path(image)
    return path.is_file() and bool(path.read_text(encoding="utf-8").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/plushie"))
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    if not 0.01 <= args.fraction < 0.5:
        raise SystemExit("--fraction must be in [0.01, 0.5)")

    dataset = args.dataset.resolve()
    train_dir = dataset / "images" / "train"
    val_dir = dataset / "images" / "val"
    images = sorted(path.resolve() for path in train_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(canonical_name(image), []).append(image)

    representatives = {
        name: next((image for image in paths if image.stem == name), paths[0])
        for name, paths in groups.items()
    }
    positives = sorted(name for name, image in representatives.items() if has_label(image))
    negatives = sorted(name for name, image in representatives.items() if not has_label(image))
    rng = random.Random(args.seed)
    selected: set[str] = set()
    for names in (positives, negatives):
        count = max(1, round(len(names) * args.fraction))
        selected.update(rng.sample(names, count))

    split_dir = dataset / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_list = split_dir / "weekend_train.txt"
    val_list = split_dir / "weekend_val.txt"
    yaml_path = dataset / "weekend_grouped_holdout.yaml"
    train_paths = [image for name, paths in groups.items() if name not in selected for image in paths]
    val_paths = [representatives[name] for name in sorted(selected)]
    val_paths.extend(sorted(path.resolve() for path in val_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))
    train_list.write_text("\n".join(map(str, sorted(train_paths))) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(map(str, val_paths)) + "\n", encoding="utf-8")
    yaml_path.write_text(
        f"train: {train_list}\nval: {val_list}\nnames:\n  0: plushie\n",
        encoding="utf-8",
    )
    stats = {
        "seed": args.seed,
        "holdout_fraction": args.fraction,
        "source_groups": len(groups),
        "selected_groups": len(selected),
        "positive_groups": len(positives),
        "negative_groups": len(negatives),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "data_yaml": str(yaml_path),
    }
    (split_dir / "weekend_holdout_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
