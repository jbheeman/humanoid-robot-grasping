from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GIB = 1024**3


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


def print_dataset_preflight(data_root: Path) -> tuple[int, int, int, int]:
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
    return train_images, val_images, train_labels, val_labels


def total_ram_gib() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return (page_size * pages) / GIB


def estimate_rgb_cache_gib(image_count: int, imgsz: int) -> float:
    # Ultralytics RAM cache stores decoded image arrays, not compressed JPG bytes.
    return (image_count * imgsz * imgsz * 3) / GIB


def resolve_cache(cache: str, image_count: int, imgsz: int, ram_reserve_gib: float, force_ram_cache: bool) -> str | bool:
    normalized = cache.strip().lower()
    if normalized in {"false", "0", "none", "off", "no"}:
        return False
    if normalized in {"true", "1", "yes", "on"}:
        normalized = "ram"
    if normalized not in {"auto", "ram"}:
        return normalized

    total_gib = total_ram_gib()
    estimated_cache_gib = estimate_rgb_cache_gib(image_count, imgsz)
    overhead_gib = 16.0
    if total_gib is None:
        print("Resource plan: total RAM unknown; using disk cache instead of RAM cache.")
        return "disk"

    ram_budget_gib = max(total_gib - ram_reserve_gib, 0.0)
    if estimated_cache_gib + overhead_gib <= ram_budget_gib:
        print(
            "Resource plan: using RAM cache "
            f"(estimated decoded cache {estimated_cache_gib:.1f} GiB, "
            f"budget {ram_budget_gib:.1f} GiB)."
        )
        return "ram"

    if normalized == "ram" and force_ram_cache:
        print(
            "Resource plan: forced RAM cache despite budget "
            f"(estimated decoded cache {estimated_cache_gib:.1f} GiB + "
            f"{overhead_gib:.1f} GiB overhead, budget {ram_budget_gib:.1f} GiB)."
        )
        return "ram"

    print(
        "Resource plan: using disk cache to avoid swap "
        f"(estimated decoded cache {estimated_cache_gib:.1f} GiB + "
        f"{overhead_gib:.1f} GiB overhead exceeds {ram_budget_gib:.1f} GiB RAM budget)."
    )
    return "disk"


def resolve_workers(workers: str, cpu_reserve_percent: float) -> int:
    normalized = workers.strip().lower()
    if normalized != "auto":
        return int(workers)

    cpus = os.cpu_count() or 1
    usable_fraction = max(0.05, min(1.0, (100.0 - cpu_reserve_percent) / 100.0))
    return max(1, math.floor(cpus * usable_fraction))


def resolve_batch(batch: str, allow_autobatch: bool) -> int | str:
    normalized = batch.strip().lower()
    if normalized in {"auto", "-1"}:
        if not allow_autobatch:
            raise SystemExit("BATCH=-1/auto is disabled by default because AutoBatch can probe unsafe sizes. Set ALLOW_AUTOBATCH=1 to override.")
        return -1
    return int(batch) if batch.lstrip("-").isdigit() else batch


def write_runtime_dataset_yaml(source_yaml: Path, data_root: Path, project: Path, name: str) -> Path:
    with source_yaml.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}

    runtime_yaml = project / f"{name}_dataset.yaml"
    runtime = {
        "path": str(data_root),
        "train": source.get("train", "images/train"),
        "val": source.get("val", "images/val"),
        "names": source.get("names", {0: "plushie"}),
    }
    if "test" in source:
        runtime["test"] = source["test"]

    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime, handle, sort_keys=False)
    return runtime_yaml


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
    parser.add_argument("--imgsz", type=int, default=env_int("IMGSZ", 1280))
    parser.add_argument("--batch", default=env_str("BATCH", "16"))
    parser.add_argument("--device", default=env_str("DEVICE", "0"))
    parser.add_argument("--workers", default=env_str("WORKERS", "auto"))
    parser.add_argument("--patience", type=int, default=env_int("PATIENCE", 40))
    parser.add_argument("--save-period", type=int, default=env_int("SAVE_PERIOD", 5))
    parser.add_argument("--cache", default=env_str("CACHE", "auto"))
    parser.add_argument("--ram-reserve-gib", type=float, default=env_float("RAM_RESERVE_GB", 16.0))
    parser.add_argument("--cpu-reserve-percent", type=float, default=env_float("CPU_RESERVE_PERCENT", 50.0))
    parser.add_argument("--force-ram-cache", action=argparse.BooleanOptionalAction, default=env_bool("FORCE_RAM_CACHE", False))
    parser.add_argument("--allow-autobatch", action=argparse.BooleanOptionalAction, default=env_bool("ALLOW_AUTOBATCH", False))
    parser.add_argument("--optimizer", default=env_str("OPTIMIZER", "auto"))
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=env_bool("COS_LR", True))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=env_bool("AMP", True))
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=env_bool("PLOTS", True))
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=env_bool("EXIST_OK", True))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=env_bool("RESUME", False))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=env_bool("DRY_RUN", False))
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

    train_images, _, _, _ = print_dataset_preflight(data_root)
    project.mkdir(parents=True, exist_ok=True)
    runtime_data_path = write_runtime_dataset_yaml(data_path, data_root, project, args.name)
    cache = resolve_cache(args.cache, train_images, args.imgsz, args.ram_reserve_gib, args.force_ram_cache)
    workers = resolve_workers(args.workers, args.cpu_reserve_percent)
    batch = resolve_batch(args.batch, args.allow_autobatch)

    print("Training resource plan:")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {batch}")
    print(f"  workers: {workers}")
    print(f"  cache: {cache}")
    print(f"  RAM reserve: {args.ram_reserve_gib:.1f} GiB")
    print(f"  CPU reserve: {args.cpu_reserve_percent:.1f}%")

    if args.dry_run:
        print("Dry run requested; exiting before YOLO model load/train.")
        return

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
            data=str(runtime_data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            project=str(project),
            name=args.name,
            device=args.device or None,
            workers=workers,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
            cache=cache,
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
