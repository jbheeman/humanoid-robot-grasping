from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from tqdm import tqdm


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}2017/{file_name}"
OPEN_IMAGES_CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OPEN_IMAGES_TRAIN_BBOX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
OPEN_IMAGES_VAL_BBOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
OPEN_IMAGES_IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"
IMAGE_SUFFIX = ".jpg"
NEGATIVE_CATEGORIES = {
    "backpack",
    "handbag",
    "suitcase",
    "dog",
    "cat",
    "sports ball",
    "chair",
    "couch",
    "bed",
    "remote",
    "book",
}
OPEN_IMAGES_POSITIVE_CATEGORIES = {"Teddy bear"}
OPEN_IMAGES_NEGATIVE_CATEGORIES = {
    "Backpack",
    "Bagel",
    "Bear",
    "Brown bear",
    "Cat",
    "Dog",
    "Dog bed",
    "Doll",
    "Handbag",
    "Luggage and bags",
    "Pillow",
    "Plastic bag",
    "Polar bear",
    "Suitcase",
    "Toy",
}
ROBOFLOW_CANDIDATES = [
    {
        "name": "soft_toy",
        "workspace": "felipe-guimaraes",
        "project": "soft-toy",
        "version": "1",
        "url": "https://universe.roboflow.com/felipe-guimaraes/soft-toy/dataset/1",
    },
    {
        "name": "stuffed_animals",
        "workspace": "babson-college-tvbix",
        "project": "stuffed-animals-d4fei",
        "version": "1",
        "url": "https://universe.roboflow.com/babson-college-tvbix/stuffed-animals-d4fei",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_curl(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--continue-at",
            "-",
            "--output",
            str(output),
            url,
        ],
        check=True,
    )


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extract_complete"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coco_yolo_line(annotation: dict[str, Any], image: dict[str, Any]) -> str:
    x, y, w, h = annotation["bbox"]
    width = image["width"]
    height = image["height"]
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return f"0 {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def download_image(url: str, output: Path) -> tuple[bool, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return True, str(output)
    try:
        urlretrieve(url, output)
        return True, str(output)
    except (HTTPError, URLError, TimeoutError) as exc:
        if output.exists():
            output.unlink()
        return False, f"{url}: {exc}"


def yolo_line_from_open_images(row: dict[str, str]) -> str:
    xmin = float(row["XMin"])
    xmax = float(row["XMax"])
    ymin = float(row["YMin"])
    ymax = float(row["YMax"])
    width = xmax - xmin
    height = ymax - ymin
    cx = xmin + width / 2.0
    cy = ymin + height / 2.0
    return f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def select_coco_records(
    instances: dict[str, Any],
    positive_category: str,
    negative_per_positive: float,
    seed: int,
    max_positive: int | None,
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]], bool]], dict[str, Any]]:
    categories = {category["id"]: category["name"] for category in instances["categories"]}
    category_ids = {name: category_id for category_id, name in categories.items()}
    positive_category_id = category_ids[positive_category]
    negative_category_ids = {category_ids[name] for name in NEGATIVE_CATEGORIES if name in category_ids}

    images = {image["id"]: image for image in instances["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    positive_annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    negative_candidate_ids: set[int] = set()

    for annotation in instances["annotations"]:
        image_id = annotation["image_id"]
        annotations_by_image.setdefault(image_id, []).append(annotation)
        category_id = annotation["category_id"]
        if category_id == positive_category_id:
            positive_annotations_by_image.setdefault(image_id, []).append(annotation)
        elif category_id in negative_category_ids:
            negative_candidate_ids.add(image_id)

    positive_image_ids = sorted(positive_annotations_by_image)
    if max_positive is not None:
        random.Random(seed).shuffle(positive_image_ids)
        positive_image_ids = sorted(positive_image_ids[:max_positive])

    positive_id_set = set(positive_image_ids)
    negative_image_ids = sorted(negative_candidate_ids - set(positive_annotations_by_image))
    negative_count = int(len(positive_image_ids) * negative_per_positive)
    random.Random(seed).shuffle(negative_image_ids)
    negative_image_ids = sorted(negative_image_ids[:negative_count])

    records = [
        (images[image_id], positive_annotations_by_image[image_id], True)
        for image_id in positive_image_ids
    ]
    records.extend((images[image_id], [], False) for image_id in negative_image_ids)

    summary = {
        "positive_category": positive_category,
        "positive_images": len(positive_image_ids),
        "negative_images": len(negative_image_ids),
        "negative_categories": sorted(NEGATIVE_CATEGORIES),
    }
    return records, summary


def install_coco_split(
    annotations_dir: Path,
    split: str,
    output_root: Path,
    workers: int,
    negative_per_positive: float,
    seed: int,
    max_positive: int | None,
) -> dict[str, Any]:
    instances = load_json(annotations_dir / "annotations" / f"instances_{split}2017.json")
    records, summary = select_coco_records(
        instances=instances,
        positive_category="teddy bear",
        negative_per_positive=negative_per_positive,
        seed=seed,
        max_positive=max_positive,
    )

    target_split = "train" if split == "train" else "val"
    image_dir = output_root / "images" / target_split
    label_dir = output_root / "labels" / target_split
    downloads = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for image, annotations, is_positive in records:
            stem = f"coco_{split}2017_{Path(image['file_name']).stem}"
            image_path = image_dir / f"{stem}{IMAGE_SUFFIX}"
            label_path = label_dir / f"{stem}.txt"
            lines = [coco_yolo_line(annotation, image) for annotation in annotations]
            write_label(label_path, lines)

            url = COCO_IMAGE_URL.format(split=split, file_name=image["file_name"])
            downloads.append((executor.submit(download_image, url, image_path), is_positive))

        failures = []
        positive_images = 0
        negative_images = 0
        for future, is_positive in tqdm(downloads, desc=f"Downloading COCO {split}2017", unit="img"):
            ok, message = future.result()
            if ok:
                positive_images += int(is_positive)
                negative_images += int(not is_positive)
            else:
                failures.append(message)

    summary.update(
        {
            "split": split,
            "target_split": target_split,
            "downloaded_positive_images": positive_images,
            "downloaded_negative_images": negative_images,
            "failures": failures[:20],
            "failure_count": len(failures),
        }
    )
    return summary


def load_open_images_classes(path: Path) -> dict[str, str]:
    run_curl(OPEN_IMAGES_CLASS_URL, path)
    class_names: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for label_id, label_name in reader:
            class_names[label_name] = label_id
    return class_names


def select_open_images_records(
    bbox_csv_path: Path,
    positive_label_ids: set[str],
    negative_label_ids: set[str],
    negative_per_positive: float,
    seed: int,
    max_positive: int | None,
) -> tuple[list[tuple[str, list[str], bool]], dict[str, Any]]:
    positive_lines_by_image: dict[str, list[str]] = {}
    negative_image_ids: set[str] = set()

    with bbox_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc=f"Scanning {bbox_csv_path.name}", unit="row"):
            label_id = row["LabelName"]
            image_id = row["ImageID"]
            if label_id in positive_label_ids:
                positive_lines_by_image.setdefault(image_id, []).append(yolo_line_from_open_images(row))
            elif label_id in negative_label_ids:
                negative_image_ids.add(image_id)

    positive_image_ids = sorted(positive_lines_by_image)
    if max_positive is not None:
        random.Random(seed).shuffle(positive_image_ids)
        positive_image_ids = sorted(positive_image_ids[:max_positive])
        positive_lines_by_image = {
            image_id: positive_lines_by_image[image_id]
            for image_id in positive_image_ids
        }

    negative_image_ids = sorted(negative_image_ids - set(positive_lines_by_image))
    negative_count = int(len(positive_lines_by_image) * negative_per_positive)
    random.Random(seed).shuffle(negative_image_ids)
    negative_image_ids = sorted(negative_image_ids[:negative_count])

    records = [(image_id, lines, True) for image_id, lines in positive_lines_by_image.items()]
    records.extend((image_id, [], False) for image_id in negative_image_ids)
    summary = {
        "positive_images": len(positive_lines_by_image),
        "negative_images": len(negative_image_ids),
        "positive_categories": sorted(OPEN_IMAGES_POSITIVE_CATEGORIES),
        "negative_categories": sorted(OPEN_IMAGES_NEGATIVE_CATEGORIES),
    }
    return records, summary


def install_open_images_split(
    bbox_csv_path: Path,
    source_split: str,
    target_split: str,
    output_root: Path,
    workers: int,
    negative_per_positive: float,
    seed: int,
    max_positive: int | None,
    class_names: dict[str, str],
) -> dict[str, Any]:
    missing_positive = sorted(OPEN_IMAGES_POSITIVE_CATEGORIES - set(class_names))
    missing_negative = sorted(OPEN_IMAGES_NEGATIVE_CATEGORIES - set(class_names))
    if missing_positive:
        raise SystemExit(f"Missing Open Images positive classes: {missing_positive}")
    if missing_negative:
        print(f"Warning: missing Open Images negative classes: {missing_negative}")

    records, summary = select_open_images_records(
        bbox_csv_path=bbox_csv_path,
        positive_label_ids={class_names[name] for name in OPEN_IMAGES_POSITIVE_CATEGORIES},
        negative_label_ids={class_names[name] for name in OPEN_IMAGES_NEGATIVE_CATEGORIES if name in class_names},
        negative_per_positive=negative_per_positive,
        seed=seed,
        max_positive=max_positive,
    )

    image_dir = output_root / "images" / target_split
    label_dir = output_root / "labels" / target_split
    downloads = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for image_id, lines, is_positive in records:
            stem = f"openimages_{source_split}_{image_id}"
            image_path = image_dir / f"{stem}{IMAGE_SUFFIX}"
            label_path = label_dir / f"{stem}.txt"
            write_label(label_path, lines)
            url = OPEN_IMAGES_IMAGE_URL.format(split=source_split, image_id=image_id)
            downloads.append((executor.submit(download_image, url, image_path), is_positive))

        failures = []
        positive_images = 0
        negative_images = 0
        for future, is_positive in tqdm(downloads, desc=f"Downloading Open Images {source_split}", unit="img"):
            ok, message = future.result()
            if ok:
                positive_images += int(is_positive)
                negative_images += int(not is_positive)
            else:
                failures.append(message)

    summary.update(
        {
            "source_split": source_split,
            "target_split": target_split,
            "downloaded_positive_images": positive_images,
            "downloaded_negative_images": negative_images,
            "failure_count": len(failures),
            "failures": failures[:20],
        }
    )
    return summary


def install_open_images(
    root: Path,
    output_root: Path,
    workers: int,
    negative_per_positive: float,
    seed: int,
    max_train_positive: int | None,
    max_val_positive: int | None,
) -> dict[str, Any]:
    source_root = root / "data" / "sources" / "openimages"
    source_root.mkdir(parents=True, exist_ok=True)
    class_csv = source_root / "oidv7-class-descriptions-boxable.csv"
    train_bbox = source_root / "oidv6-train-annotations-bbox.csv"
    val_bbox = source_root / "validation-annotations-bbox.csv"

    class_names = load_open_images_classes(class_csv)
    run_curl(OPEN_IMAGES_TRAIN_BBOX_URL, train_bbox)
    run_curl(OPEN_IMAGES_VAL_BBOX_URL, val_bbox)

    splits = [
        install_open_images_split(
            train_bbox,
            "train",
            "train",
            output_root,
            workers=workers,
            negative_per_positive=negative_per_positive,
            seed=seed,
            max_positive=max_train_positive,
            class_names=class_names,
        ),
        install_open_images_split(
            val_bbox,
            "validation",
            "val",
            output_root,
            workers=max(workers // 2, 1),
            negative_per_positive=negative_per_positive,
            seed=seed,
            max_positive=max_val_positive,
            class_names=class_names,
        ),
    ]
    return {
        "installed_source": "openimages_teddy_bear_as_plushie",
        "class_mapping": {"openimages:Teddy bear": "plushie"},
        "splits": splits,
    }


def roboflow_download_url(candidate: dict[str, str], api_key: str) -> str:
    return (
        f"https://universe.roboflow.com/{candidate['workspace']}/{candidate['project']}"
        f"/dataset/{candidate['version']}/download/yolov8?api_key={api_key}"
    )


def merge_roboflow_export(export_dir: Path, output_root: Path, prefix: str) -> dict[str, Any]:
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    counts: dict[str, int] = {"train": 0, "val": 0}

    for source_split, target_split in split_map.items():
        images_dir = export_dir / source_split / "images"
        labels_dir = export_dir / source_split / "labels"
        if not images_dir.exists():
            continue

        for image_path in tqdm(list(images_dir.glob("*")), desc=f"Merging Roboflow {prefix} {source_split}", unit="img"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = f"roboflow_{prefix}_{source_split}_{image_path.stem}"
            target_image = output_root / "images" / target_split / f"{stem}{image_path.suffix.lower()}"
            target_label = output_root / "labels" / target_split / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, target_image)

            source_label = labels_dir / f"{image_path.stem}.txt"
            normalized_lines = []
            if source_label.exists():
                for line in source_label.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        normalized_lines.append(" ".join(["0", *parts[1:5]]))
            write_label(target_label, normalized_lines)
            counts[target_split] += 1

    return counts


def install_roboflow(root: Path, output_root: Path, api_key: str | None) -> dict[str, Any]:
    source_root = root / "data" / "sources" / "roboflow"
    source_root.mkdir(parents=True, exist_ok=True)

    if not api_key:
        return {
            "installed_sources": [],
            "skipped": [
                {
                    "name": candidate["name"],
                    "reason": "ROBOFLOW_API_KEY is not set",
                    "url": candidate["url"],
                }
                for candidate in ROBOFLOW_CANDIDATES
            ],
        }

    installed = []
    skipped = []
    for candidate in ROBOFLOW_CANDIDATES:
        zip_path = source_root / f"{candidate['name']}.zip"
        extract_dir = source_root / candidate["name"]
        try:
            run_curl(roboflow_download_url(candidate, api_key), zip_path)
            extract_zip(zip_path, extract_dir)
            counts = merge_roboflow_export(extract_dir, output_root, candidate["name"])
            installed.append({"name": candidate["name"], "counts": counts, "url": candidate["url"]})
        except Exception as exc:
            skipped.append({"name": candidate["name"], "reason": str(exc), "url": candidate["url"]})

    return {
        "installed_sources": installed,
        "skipped": skipped,
    }


def copy_yaml(root: Path) -> None:
    source = root / "data" / "plushie" / "plushie.yaml"
    target = root / "data" / "plushie" / "sources.yaml"
    shutil.copyfile(source, target)


def cleanup_orphan_labels(output_root: Path) -> dict[str, int]:
    removed: dict[str, int] = {"train": 0, "val": 0}
    for split in ("train", "val"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        if not label_dir.exists():
            continue
        image_stems = {
            image_path.stem
            for image_path in image_dir.glob("*")
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        for label_path in label_dir.glob("*.txt"):
            if label_path.stem not in image_stems:
                label_path.unlink()
                removed[split] += 1
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install public plushie training sources into data/plushie.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--negative-per-positive", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-positive", type=int, default=0, help="0 means all COCO train teddy-bear images.")
    parser.add_argument("--max-val-positive", type=int, default=0, help="0 means all COCO val teddy-bear images.")
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--skip-open-images", action="store_true")
    parser.add_argument("--skip-roboflow", action="store_true")
    parser.add_argument("--max-openimages-train-positive", type=int, default=0)
    parser.add_argument("--max-openimages-val-positive", type=int, default=0)
    parser.add_argument("--roboflow-api-key", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root()
    source_root = root / "data" / "sources" / "coco2017"
    plushie_root = root / "data" / "plushie"
    zip_path = source_root / "annotations_trainval2017.zip"

    coco_summaries = []
    if not args.skip_coco:
        run_curl(COCO_ANNOTATIONS_URL, zip_path)
        extract_zip(zip_path, source_root)
        coco_summaries = [
            install_coco_split(
                source_root,
                "train",
                plushie_root,
                workers=args.workers,
                negative_per_positive=args.negative_per_positive,
                seed=args.seed,
                max_positive=args.max_train_positive or None,
            ),
            install_coco_split(
                source_root,
                "val",
                plushie_root,
                workers=max(args.workers // 2, 1),
                negative_per_positive=args.negative_per_positive,
                seed=args.seed,
                max_positive=args.max_val_positive or None,
            ),
        ]

    open_images_summary = None
    if not args.skip_open_images:
        open_images_summary = install_open_images(
            root=root,
            output_root=plushie_root,
            workers=args.workers,
            negative_per_positive=args.negative_per_positive,
            seed=args.seed,
            max_train_positive=args.max_openimages_train_positive or None,
            max_val_positive=args.max_openimages_val_positive or None,
        )

    roboflow_summary = None
    if not args.skip_roboflow:
        roboflow_summary = install_roboflow(
            root=root,
            output_root=plushie_root,
            api_key=args.roboflow_api_key or None,
        )

    copy_yaml(root)
    orphan_labels_removed = cleanup_orphan_labels(plushie_root)
    manifest = {
        "installed_sources": [
            source
            for source in [
                None if args.skip_coco else "coco2017_teddy_bear_as_plushie",
                None if open_images_summary is None else open_images_summary["installed_source"],
            ]
            if source is not None
        ],
        "class_mapping": {
            "coco:teddy bear": "plushie",
            "openimages:Teddy bear": "plushie",
        },
        "coco": {"splits": coco_summaries},
        "open_images": open_images_summary,
        "roboflow": roboflow_summary,
        "orphan_labels_removed": orphan_labels_removed,
    }
    manifest_path = plushie_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
