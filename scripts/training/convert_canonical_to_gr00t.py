#!/usr/bin/env python3
"""Convert the canonical G1 bunny HDF5 store to GR00T-flavored LeRobot v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30
INSTRUCTION = "Block the moving rabbit plush with the right palm."
VIDEO_KEY = "observation.images.ego_view"
ANNOTATION_COLUMN = "annotation.human.task_description"
RIGHT_POSE17 = slice(6, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_6d(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    first = np.stack((cy * cp, sy * cp, -sp), axis=-1)
    second = np.stack(
        (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        axis=-1,
    )
    return np.concatenate((first, second), axis=-1)


def right_pose9(pose17: np.ndarray) -> np.ndarray:
    right = np.asarray(pose17[:, RIGHT_POSE17], dtype=np.float32)
    if right.ndim != 2 or right.shape[1] != 6:
        raise ValueError(f"expected [T, 6] right EEF pose, got {right.shape}")
    return np.concatenate((right[:, :3], rotation_6d(right[:, 3:6])), axis=1).astype(
        np.float32
    )


def write_video(frames: h5py.Dataset, length: int, destination: Path) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"invalid image dataset shape {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for start in range(0, length, 16):
            batch = np.asarray(frames[start : min(length, start + 16)], dtype=np.uint8)
            process.stdin.write(np.ascontiguousarray(batch).tobytes())
        process.stdin.close()
        result = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}: {destination}")


def fixed_list(values: np.ndarray) -> pa.Array:
    width = int(values.shape[1])
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), width
    )


def write_episode(
    source: Path,
    dataset_root: Path,
    source_name: str,
    source_episode: str,
    episode_index: int,
    global_index: int,
) -> tuple[dict[str, Any], int]:
    lag = 3 if source_name == "xr_teleoperate" else 1
    with h5py.File(source, "r") as root:
        pose9 = right_pose9(np.asarray(root["observations/ee_qpos"][:], dtype=np.float32))
        usable = len(pose9) - lag
        if usable < 17:
            raise ValueError(f"{source}: only {usable} usable frames for a 16-step horizon")
        state = pose9[:usable]
        action = pose9[lag:]
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{source}: non-finite state/action")
        timestamps = np.asarray(root["timestamp"][:usable], dtype=np.float64)
        timestamps = (timestamps - timestamps[0]).astype(np.float32)
        images = root["observations/images/cam_left_high"]
        video_path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )
        write_video(images, usable, video_path)

    local_indices = np.arange(usable, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": fixed_list(state),
            "action": fixed_list(action),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            ANNOTATION_COLUMN: pa.array(np.zeros(usable, dtype=np.int64)),
            "task_index": pa.array(np.zeros(usable, dtype=np.int64)),
            "episode_index": pa.array(
                np.full(usable, episode_index, dtype=np.int64)
            ),
            "frame_index": pa.array(local_indices),
            "index": pa.array(local_indices + global_index),
            "next.reward": pa.array(np.zeros(usable, dtype=np.float32)),
            "next.done": pa.array(
                np.arange(usable, dtype=np.int64) == usable - 1, type=pa.bool_()
            ),
        }
    )
    parquet_path = (
        dataset_root
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd")
    return (
        {
            "episode_index": episode_index,
            "tasks": [INSTRUCTION],
            "length": usable,
            "source": source_name,
            "source_episode": source_episode,
            "source_path": str(source),
            "label_lag_frames": lag,
        },
        global_index + usable,
    )


def info_payload(episode_count: int, frame_count: int) -> dict[str, Any]:
    vector_feature = {
        "dtype": "float32",
        "shape": [9],
        "names": [
            "right_palm_x",
            "right_palm_y",
            "right_palm_z",
            "right_palm_rot6d_0",
            "right_palm_rot6d_1",
            "right_palm_rot6d_2",
            "right_palm_rot6d_3",
            "right_palm_rot6d_4",
            "right_palm_rot6d_5",
        ],
    }
    scalar_float = {"dtype": "float32", "shape": [1], "names": None}
    scalar_int = {"dtype": "int64", "shape": [1], "names": None}
    return {
        "codebase_version": "v2.1",
        "robot_type": "G1_RIGHT_PALM",
        "total_episodes": episode_count,
        "total_frames": frame_count,
        "total_tasks": 1,
        "total_videos": episode_count,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{episode_count}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": vector_feature,
            "action": vector_feature,
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": scalar_float,
            ANNOTATION_COLUMN: scalar_int,
            "task_index": scalar_int,
            "episode_index": scalar_int,
            "frame_index": scalar_int,
            "index": scalar_int,
            "next.reward": scalar_float,
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_dataset(
    records: list[dict[str, Any]],
    dataset_root: Path,
    store_root: Path,
) -> dict[str, Any]:
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    global_index = 0
    for episode_index, record in enumerate(records):
        source = store_root / record["path"]
        print(
            f"CONVERT dataset={dataset_root.name} episode={episode_index + 1}/"
            f"{len(records)} source={source}",
            flush=True,
        )
        converted, global_index = write_episode(
            source,
            dataset_root,
            record["source"],
            Path(record["path"]).stem,
            episode_index,
            global_index,
        )
        episodes.append(converted)
    meta = dataset_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    write_json(meta / "info.json", info_payload(len(episodes), global_index))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": INSTRUCTION}) + "\n"
    )
    with (meta / "episodes.jsonl").open("w") as stream:
        for episode in episodes:
            stream.write(
                json.dumps(
                    {
                        "episode_index": episode["episode_index"],
                        "tasks": episode["tasks"],
                        "length": episode["length"],
                    }
                )
                + "\n"
            )
    with (meta / "source_episodes.jsonl").open("w") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, sort_keys=True) + "\n")
    write_json(
        meta / "modality.json",
        {
            "state": {"right_palm_eef": {"start": 0, "end": 9}},
            "action": {"right_palm_eef": {"start": 0, "end": 9}},
            "video": {"ego_view": {"original_key": VIDEO_KEY}},
            "annotation": {"human.task_description": {}},
        },
    )
    return {
        "dataset_root": str(dataset_root),
        "episodes": len(episodes),
        "frames": global_index,
    }


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"{dataset_root}: episode count mismatch")
    frames = 0
    for episode in episodes:
        index = episode["episode_index"]
        parquet = (
            dataset_root / "data/chunk-000" / f"episode_{index:06d}.parquet"
        )
        video = (
            dataset_root
            / "videos/chunk-000"
            / VIDEO_KEY
            / f"episode_{index:06d}.mp4"
        )
        table = pq.read_table(parquet)
        if table.num_rows != episode["length"]:
            raise ValueError(f"{parquet}: row count mismatch")
        for key in ("observation.state", "action"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (table.num_rows, 9) or not np.isfinite(values).all():
                raise ValueError(f"{parquet}: invalid {key}")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if stream["codec_name"] != "h264" or stream["avg_frame_rate"] != "30/1":
            raise ValueError(f"{video}: invalid video contract {stream}")
        if int(stream["nb_read_frames"]) != table.num_rows:
            raise ValueError(f"{video}: frame count mismatch {stream}")
        frames += table.num_rows
    if frames != info["total_frames"]:
        raise ValueError(f"{dataset_root}: total frame count mismatch")
    return {
        "dataset_root": str(dataset_root),
        "episodes": len(episodes),
        "frames": frames,
        "info_sha256": sha256(dataset_root / "meta/info.json"),
        "modality_sha256": sha256(dataset_root / "meta/modality.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "real", "sim"), default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in manifest["episodes"]:
        source = "real" if record["source"] == "xr_teleoperate" else "sim"
        if args.only != "all" and source != args.only:
            continue
        grouped.setdefault((source, record["split"]), []).append(record)

    summaries = []
    for (source, split), records in sorted(grouped.items()):
        destination = args.output_root / f"{source}_{split}"
        if not args.validate_only:
            summaries.append(build_dataset(records, destination, args.store_root))
        summaries.append(validate_dataset(destination))

    valid_windows = {
        item["dataset_root"]: item["frames"]
        for item in summaries
        if "frames" in item and item["dataset_root"].endswith("_train")
    }
    real_frames = next(
        (frames for path, frames in valid_windows.items() if path.endswith("real_train")),
        None,
    )
    sim_frames = next(
        (frames for path, frames in valid_windows.items() if path.endswith("sim_train")),
        None,
    )
    alpha = None
    if real_frames and sim_frames and real_frames != sim_frames:
        alpha = math.log(3.0) / math.log(real_frames / sim_frames)
    report = {
        "schema_version": 1,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256(args.manifest),
        "datasets": summaries,
        "target_real_sampling_mass": 0.75,
        "target_sim_sampling_mass": 0.25,
        "ds_weights_alpha": alpha,
    }
    write_json(args.output_root / "CONVERSION_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
