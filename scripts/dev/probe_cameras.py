#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

cv2 = None


def backend_name(api_id: int) -> str:
    try:
        name = cv2.videoio_registry.getBackendName(api_id)
    except Exception:
        return str(api_id)
    return f"{name} ({api_id})"


def available_backends() -> list[str]:
    try:
        return [backend_name(api_id) for api_id in cv2.videoio_registry.getBackends()]
    except Exception as exc:
        return [f"Could not read backend registry: {exc}"]


def parse_indices(value: str) -> list[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            indices.update(range(start, end + step, step))
        else:
            indices.add(int(part))
    return sorted(indices)


def print_header() -> None:
    print("OpenCV camera probe")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"OpenCV: {cv2.__version__}")
    print("Available video backends:")
    for backend in available_backends():
        print(f"  - {backend}")
    print()


def read_frames(cap: cv2.VideoCapture, count: int) -> tuple[int, tuple[int, ...] | None]:
    reads_ok = 0
    last_shape: tuple[int, ...] | None = None
    for _ in range(count):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        reads_ok += 1
        last_shape = tuple(frame.shape)
    return reads_ok, last_shape


def probe_camera(index: int, backend: int, read_count: int) -> None:
    backend_label = "default backend" if backend == cv2.CAP_ANY else backend_name(backend)
    print(f"Camera index {index} using {backend_label}")

    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            print("  opened: no")
            print("  result: unavailable or blocked")
            return

        print("  opened: yes")
        reads_ok, frame_shape = read_frames(cap, read_count)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))

        print(f"  backend: {backend_name(backend_id)}")
        print(f"  configured size: {width:.0f}x{height:.0f}")
        print(f"  reported fps: {fps:.2f}")
        print(f"  frame reads: {reads_ok}/{read_count}")
        if frame_shape is None:
            print("  last frame shape: none")
        else:
            print(f"  last frame shape: {frame_shape}")
        print("  result: usable" if reads_ok > 0 else "  result: opened but did not return frames")
    finally:
        cap.release()


def probe_indices(indices: Iterable[int], backend: int, read_count: int) -> None:
    for index in indices:
        probe_camera(index, backend, read_count)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe local OpenCV camera indexes and print diagnostics to stdout."
    )
    parser.add_argument(
        "--indices",
        default="0-5",
        help="Comma-separated camera indexes or ranges to test, for example: 0,1,3 or 0-5.",
    )
    parser.add_argument(
        "--read-frames",
        type=int,
        default=5,
        help="Number of frames to try reading from each opened camera.",
    )
    parser.add_argument(
        "--backend",
        choices=("any", "avfoundation", "v4l2", "dshow", "msmf"),
        default="any",
        help="OpenCV video backend to request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for the OpenCV import and camera probe before giving up.",
    )
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def run_probe(args: argparse.Namespace) -> None:
    global cv2
    cv2 = importlib.import_module("cv2")

    backends = {
        "any": cv2.CAP_ANY,
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    }

    if args.read_frames < 1:
        raise SystemExit("--read-frames must be at least 1")

    indices = parse_indices(args.indices)
    if not indices:
        raise SystemExit("--indices did not contain any camera indexes")

    print_header()
    probe_indices(indices, backends[args.backend], args.read_frames)


def run_worker_with_timeout(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--indices",
        args.indices,
        "--read-frames",
        str(args.read_frames),
        "--backend",
        args.backend,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        print("OpenCV camera probe")
        print(f"result: timed out after {args.timeout:.1f}s")
        if exc.stdout:
            print()
            print(exc.stdout, end="")
        if exc.stderr:
            print()
            print("stderr:")
            print(exc.stderr, end="")
        print()
        print("Next checks:")
        print("  - Confirm the virtualenv OpenCV install can import: python -c 'import cv2; print(cv2.__version__)'")
        print("  - On macOS, confirm Terminal or your IDE has Camera permission.")
        print("  - Try a smaller probe: python scripts/dev/probe_cameras.py --indices 0 --backend avfoundation")
        return 124

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print("stderr:", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def main() -> None:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")
    if args._worker:
        run_probe(args)
        return

    raise SystemExit(run_worker_with_timeout(args))


if __name__ == "__main__":
    main()
