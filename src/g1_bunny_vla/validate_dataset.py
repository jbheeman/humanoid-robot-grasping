"""Validate captured HDF5 episodes before RLDS conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contract import CAMERA_NAMES, DatasetContract


def validate_episode(path: Path, fps_tolerance: float = 1.0) -> list[str]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("validation requires h5py") from exc
    errors: list[str] = []
    contract = DatasetContract()
    with h5py.File(path, "r") as root:
        required = {
            "observations/qpos": contract.qpos_dim,
            "observations/qvel": contract.qpos_dim,
            "observations/ee_qpos": contract.ee_dim,
            "action": contract.qpos_dim,
            "ee_action": contract.ee_dim,
        }
        lengths: dict[str, int] = {}
        for key, width in required.items():
            if key not in root:
                errors.append(f"missing {key}")
                continue
            values = root[key]
            lengths[key] = values.shape[0]
            if values.ndim != 2 or values.shape[1] != width:
                errors.append(f"{key} expected [T,{width}], got {values.shape}")
            elif not np.isfinite(values[:]).all():
                errors.append(f"{key} contains non-finite values")
        for camera in CAMERA_NAMES:
            key = f"observations/images/{camera}"
            if key not in root:
                errors.append(f"missing {key}")
                continue
            images = root[key]
            lengths[key] = images.shape[0]
            expected_tail = (contract.image_height, contract.image_width, 3)
            if images.shape[1:] != expected_tail or images.dtype != np.uint8:
                errors.append(f"{key} expected [T,{expected_tail}] uint8, got {images.shape} {images.dtype}")
        if lengths and len(set(lengths.values())) != 1:
            errors.append(f"inconsistent episode lengths: {lengths}")
        if "timestamp" not in root:
            errors.append("missing timestamp")
        else:
            timestamps = np.asarray(root["timestamp"][:], dtype=np.float64)
            if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
                errors.append("timestamps must be strictly increasing and contain at least two frames")
            else:
                measured_fps = 1.0 / float(np.median(np.diff(timestamps)))
                if abs(measured_fps - contract.fps) > fps_tolerance:
                    errors.append(f"median frame rate {measured_fps:.3f} Hz is outside tolerance")
        if "language_raw" not in root:
            errors.append("missing language_raw")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--fps-tolerance", type=float, default=1.0)
    args = parser.parse_args()
    paths = sorted(args.dataset_dir.glob("episode_*.hdf5"))
    if not paths:
        print(f"No episode_*.hdf5 files found in {args.dataset_dir}")
        return 2
    failed = 0
    for path in paths:
        errors = validate_episode(path, args.fps_tolerance)
        if errors:
            failed += 1
            print(f"FAIL {path.name}: " + "; ".join(errors))
        else:
            print(f"PASS {path.name}")
    print(f"Validated {len(paths)} episodes; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

