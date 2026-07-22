#!/usr/bin/env python3
"""Export an HDF5 episode as an annotated review video and contact trace."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode", type=Path)
parser.add_argument("--output-prefix", type=Path, required=True)
parser.add_argument("--camera", default="cam_left_high")
args = parser.parse_args()
args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(args.episode, "r") as episode:
    images = episode[f"observations/images/{args.camera}"]
    timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
    forces = np.asarray(episode["sim_signals/contact_force"][:, :3], dtype=np.float32)
    force_norm = np.linalg.norm(forces, axis=1)
    tangent_speed = np.asarray(episode.get("sim_signals/path_tangent_speed", np.zeros(len(images))), dtype=np.float32)
    lateral_speed = np.asarray(episode.get("sim_signals/lateral_speed", np.zeros(len(images))), dtype=np.float32)
    angular_speed = np.asarray(episode.get("sim_signals/angular_speed", np.zeros(len(images))), dtype=np.float32)
    tilt_degrees = np.asarray(episode.get("sim_signals/tilt_degrees", np.zeros(len(images))), dtype=np.float32)
    threshold = float(episode.attrs.get("contact_force_threshold_n", 0.5))
    contact = force_norm > threshold
    if "sim_signals/contacting_link" in episode:
        links = [value.decode() if isinstance(value, bytes) else str(value) for value in episode["sim_signals/contacting_link"]]
    else:
        links = [""] * len(images)
    fps = 1.0 / float(np.median(np.diff(timestamps)))
    height, width = images.shape[1:3]
    video_path = args.output_prefix.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {video_path}")
    first_contact = None
    for index, rgb in enumerate(images):
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        active = bool(contact[index])
        color = (40, 40, 240) if active else (230, 230, 230)
        label = (
            f"frame {index:03d} t={timestamps[index]:.2f}s force={force_norm[index]:.2f}N "
            f"path={tangent_speed[index]:+.2f}m/s spin={angular_speed[index]:.2f}rad/s "
            f"tilt={tilt_degrees[index]:.1f}deg"
        )
        if active:
            label += f"  RIGHT HAND <-> BUNNY CONTACT  link={links[index]}"
            if first_contact is None:
                first_contact = index
        cv2.rectangle(bgr, (0, height - 38), (width, height), (15, 15, 15), -1)
        cv2.putText(bgr, label, (10, height - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        writer.write(bgr)
        if active and index == first_contact:
            cv2.imwrite(str(args.output_prefix.with_name(args.output_prefix.name + "_first_contact").with_suffix(".jpg")), bgr)
    writer.release()

    with args.output_prefix.with_name(args.output_prefix.name + "_contact_trace").with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        output = csv.writer(stream)
        output.writerow(("frame", "timestamp_s", "force_x_n", "force_y_n", "force_z_n", "force_norm_n", "contact", "contacting_link", "path_speed_m_s", "lateral_speed_m_s", "spin_rad_s", "tilt_deg"))
        for index, (timestamp, force, norm, active, link) in enumerate(zip(timestamps, forces, force_norm, contact, links)):
            output.writerow((index, float(timestamp), *map(float, force), float(norm), int(active), link, float(tangent_speed[index]), float(lateral_speed[index]), float(angular_speed[index]), float(tilt_degrees[index])))

    contact_indices = np.flatnonzero(contact)
    summary = {
        "source_episode": str(args.episode.resolve()),
        "camera": args.camera,
        "frames": int(len(images)),
        "fps": fps,
        "pair_contact_threshold_n": threshold,
        "contact_frames": contact_indices.tolist(),
        "contacting_links": sorted({links[index] for index in contact_indices if links[index]}),
        "first_contact_frame": None if first_contact is None else int(first_contact),
        "max_pair_contact_force_n": float(force_norm.max(initial=0.0)),
        "pair_filtered_contact": bool(episode.attrs.get("pair_filtered_contact", False)),
        "object_anchored_until_contact": bool(episode.attrs.get("object_anchored_until_contact", True)),
        "launch_speed_m_s": float(episode.attrs.get("launch_speed_m_s", 0.0)),
        "launch_heading_deg": float(episode.attrs.get("launch_heading_deg", 0.0)),
        "reaction_time_s": float(episode.attrs.get("reaction_time_s", 0.0)),
        "extension_m": float(episode.attrs.get("extension_m", 0.0)),
        "actual_palm_extension_at_contact_m": float(episode.attrs.get("actual_palm_extension_at_contact_m", 0.0)),
        "pre_contact_path_speed_m_s": float(episode.attrs.get("pre_contact_path_speed_m_s", 0.0)),
        "post_contact_path_speed_m_s": float(episode.attrs.get("post_contact_path_speed_m_s", 0.0)),
        "stop_fraction": float(episode.attrs.get("stop_fraction", 0.0)),
        "counterfactual_stop_margin_m_s": float(episode.attrs.get("counterfactual_stop_margin_m_s", 0.0)),
        "sustained_stop_frames": int(episode.attrs.get("sustained_stop_frames", 0)),
        "maximum_rebound_path_speed_m_s": float(episode.attrs.get("maximum_rebound_path_speed_m_s", 0.0)),
        "maximum_angular_speed_rad_s": float(episode.attrs.get("maximum_angular_speed_rad_s", 0.0)),
        "maximum_tilt_deg": float(episode.attrs.get("maximum_tilt_deg", 0.0)),
        "table_exit": bool(episode.attrs.get("table_exit", True)),
        "fixed_base_balance_not_validated": bool(episode.attrs.get("fixed_base_balance_not_validated", True)),
    }
    args.output_prefix.with_name(args.output_prefix.name + "_summary").with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
print(json.dumps(summary, sort_keys=True))
