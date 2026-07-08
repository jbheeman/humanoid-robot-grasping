from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def count_files(path: Path, suffixes: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    files = [item for item in path.rglob("*") if item.is_file()]
    if suffixes is not None:
        files = [item for item in files if item.suffix.lower() in suffixes]
    count = 0
    for _ in tqdm(files, desc=f"Scanning {path}", unit="file"):
        count += 1
    return count


def print_dataset_preflight(data_root: Path) -> None:
    train_images = count_files(data_root / "images" / "train", IMAGE_SUFFIXES)
    val_images = count_files(data_root / "images" / "val", IMAGE_SUFFIXES)
    train_labels = count_files(data_root / "labels" / "train", {".txt"})
    val_labels = count_files(data_root / "labels" / "val", {".txt"})

    print("Dataset preflight:")
    print(f"  train images: {train_images}")
    print(f"  val images:   {val_images}")
    print(f"  train labels: {train_labels}")
    print(f"  val labels:   {val_labels}")

    if train_images == 0 or val_images == 0:
        raise SystemExit("Dataset is empty. Add labeled train/val images before training.")


def resolve_model(model: str, pretrained_dir: Path) -> tuple[str, Path | None]:
    model_path = Path(model)
    if model_path.is_absolute() or model_path.parent != Path("."):
        return str(model_path), None

    pretrained_dir.mkdir(parents=True, exist_ok=True)
    existing = pretrained_dir / model
    if existing.exists():
        return str(existing), None

    # Run from models/pretrained so Ultralytics downloads bare weight names there.
    return model, pretrained_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the plushie detector with all artifacts under models/.")
    parser.add_argument("--data", default=env_str("DATA", "data/plushie/plushie.yaml"))
    parser.add_argument("--data-root", default=env_str("DATA_ROOT", "data/plushie"))
    parser.add_argument("--model", default=env_str("MODEL", "yolo11x.pt"))
    parser.add_argument("--project", default=env_str("PROJECT", "models/plushie_detector"))
    parser.add_argument("--name", default=env_str("NAME", "yolo11x_plushie"))
    parser.add_argument("--epochs", type=int, default=env_int("EPOCHS", 160))
    parser.add_argument("--imgsz", type=int, default=env_int("IMGSZ", 960))
    parser.add_argument("--batch", default=env_str("BATCH", "-1"))
    parser.add_argument("--device", default=env_str("DEVICE", ""))
    parser.add_argument("--workers", type=int, default=env_int("WORKERS", 12))
    parser.add_argument("--patience", type=int, default=env_int("PATIENCE", 40))
    parser.add_argument("--save-period", type=int, default=env_int("SAVE_PERIOD", 5))
    parser.add_argument("--cache", default=env_str("CACHE", "ram"))
    parser.add_argument("--optimizer", default=env_str("OPTIMIZER", "auto"))
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=env_bool("COS_LR", True))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=env_bool("AMP", True))
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=env_bool("PLOTS", True))
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=env_bool("EXIST_OK", True))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=env_bool("RESUME", False))
    parser.add_argument(
        "--resume-model",
        default=env_str("RESUME_MODEL", "models/plushie_detector/yolo11x_plushie/weights/last.pt"),
    )
    return parser


def main() -> None:
    root = repo_root()
    args = build_parser().parse_args()

    data_path = (root / args.data).resolve()
    data_root = (root / args.data_root).resolve()
    project = (root / args.project).resolve()
    pretrained_dir = (root / "models" / "pretrained").resolve()

    print_dataset_preflight(data_root)
    project.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model_ref = str((root / args.resume_model).resolve())
        if not Path(model_ref).exists():
            raise SystemExit(f"Resume checkpoint not found: {model_ref}")
        resume = True
        cwd = None
    else:
        model_ref, cwd = resolve_model(args.model, pretrained_dir)
        resume = False

    previous_cwd = Path.cwd()
    try:
        if cwd is not None:
            os.chdir(cwd)

        model = YOLO(model_ref)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch,
            project=str(project),
            name=args.name,
            device=args.device or None,
            workers=args.workers,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
            cache=args.cache,
            optimizer=args.optimizer,
            cos_lr=args.cos_lr,
            amp=args.amp,
            plots=args.plots,
            exist_ok=args.exist_ok,
            resume=resume,
            verbose=True,
        )
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
