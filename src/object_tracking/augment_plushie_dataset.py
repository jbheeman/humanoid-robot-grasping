from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_yolo_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append([float(parts[0]), *[float(value) for value in parts[1:5]]])
    return labels


def write_yolo_labels(path: Path, labels: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}"
        for label in labels
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def clamp_label(label: list[float]) -> list[float] | None:
    cls_id, cx, cy, width, height = label
    x1 = max(cx - width / 2.0, 0.0)
    y1 = max(cy - height / 2.0, 0.0)
    x2 = min(cx + width / 2.0, 1.0)
    y2 = min(cy + height / 2.0, 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [cls_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def adjust_labels_for_canvas(
    labels: list[list[float]],
    source_width: int,
    source_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_width: int,
    target_height: int,
) -> list[list[float]]:
    adjusted = []
    for cls_id, cx, cy, width, height in labels:
        new_cx = (cx * source_width * scale + pad_x) / target_width
        new_cy = (cy * source_height * scale + pad_y) / target_height
        new_width = width * source_width * scale / target_width
        new_height = height * source_height * scale / target_height
        clamped = clamp_label([cls_id, new_cx, new_cy, new_width, new_height])
        if clamped is not None:
            adjusted.append(clamped)
    return adjusted


def motion_blur(image: np.ndarray, kernel_size: int, horizontal: bool) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if horizontal:
        kernel[kernel_size // 2, :] = 1.0
    else:
        kernel[:, kernel_size // 2] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_sensor_noise(image: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    noise_seed = rng.randrange(0, 2**32 - 1)
    np_rng = np.random.default_rng(noise_seed)
    noise = np_rng.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gray_floor_canvas(
    image: np.ndarray,
    labels: list[list[float]],
    rng: random.Random,
    target_width: int = 1280,
    target_height: int = 720,
) -> tuple[np.ndarray, list[list[float]]]:
    source_height, source_width = image.shape[:2]
    gray = rng.randint(105, 150)
    canvas = np.full((target_height, target_width, 3), gray, dtype=np.uint8)

    floor_y = int(target_height * rng.uniform(0.55, 0.72))
    cv2.rectangle(canvas, (0, floor_y), (target_width, target_height), (gray - 12, gray - 12, gray - 12), -1)

    base_scale = min(target_width / source_width, target_height / source_height)
    scale = base_scale * rng.uniform(0.58, 0.95)
    new_width = max(1, int(source_width * scale))
    new_height = max(1, int(source_height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    max_x = max(target_width - new_width, 0)
    max_y = max(target_height - new_height, 0)
    pad_x = rng.randint(0, max_x) if max_x else 0
    if max_y:
        pad_y = rng.randint(max(0, int(max_y * 0.35)), max_y)
    else:
        pad_y = 0

    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    adjusted = adjust_labels_for_canvas(
        labels,
        source_width=source_width,
        source_height=source_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        target_width=target_width,
        target_height=target_height,
    )
    return canvas, adjusted


def augment_image(
    image: np.ndarray,
    labels: list[list[float]],
    variant: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[list[float]], str]:
    if variant == 0:
        output = cv2.GaussianBlur(image, (5, 5), 0)
        output = jpeg_roundtrip(output, quality=rng.randint(38, 62))
        output = brightness_contrast(output, alpha=rng.uniform(0.82, 1.18), beta=rng.uniform(-18, 18))
        return output, labels, "blur_jpeg"

    if variant == 1:
        kernel = rng.choice([7, 9, 11, 13])
        output = motion_blur(image, kernel_size=kernel, horizontal=rng.choice([True, False]))
        output = add_sensor_noise(output, sigma=rng.uniform(3.0, 8.0), rng=rng)
        output = jpeg_roundtrip(output, quality=rng.randint(42, 68))
        return output, labels, "motion_noise"

    canvas, adjusted = gray_floor_canvas(image, labels, rng)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = jpeg_roundtrip(canvas, quality=rng.randint(45, 70))
    return canvas, adjusted, "g1_gray_720p"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create train-only plushie dataset augmentations.")
    parser.add_argument("--data-root", default="data/plushie")
    parser.add_argument("--aug-per-image", type=int, default=3)
    parser.add_argument("--max-source-images", type=int, default=0, help="0 means all non-augmented train images.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root()
    data_root = root / args.data_root
    image_dir = data_root / "images" / "train"
    label_dir = data_root / "labels" / "train"

    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.stem.startswith("aug_")
    ]
    if args.max_source_images:
        image_paths = image_paths[: args.max_source_images]

    rng = random.Random(args.seed)
    written = 0
    skipped = 0

    for image_path in tqdm(image_paths, desc="Augmenting train images", unit="img"):
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        for index in range(args.aug_per_image):
            variant = index % 3
            augmented, adjusted_labels, suffix = augment_image(image, labels, variant, rng)
            output_stem = f"aug_{suffix}_{image_path.stem}_{index:02d}"
            output_image = image_dir / f"{output_stem}.jpg"
            output_label = label_dir / f"{output_stem}.txt"
            if output_image.exists() and output_label.exists():
                skipped += 1
                continue

            cv2.imwrite(str(output_image), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            write_yolo_labels(output_label, adjusted_labels)
            written += 1

    print(
        {
            "source_images": len(image_paths),
            "aug_per_image": args.aug_per_image,
            "written": written,
            "skipped": skipped,
            "train_image_dir": str(image_dir),
            "train_label_dir": str(label_dir),
        }
    )


if __name__ == "__main__":
    main()
