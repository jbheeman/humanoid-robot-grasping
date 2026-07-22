#!/usr/bin/env python3
"""Prepare Unitree's pinned G1+BrainCo URDF for one semantic palm contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_UNITREE_COMMIT = "aa0f5c68b5aba347bad409e71b6430407da758d7"
PALM_CENTER_Z_M = 0.047


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descendants(root: ET.Element, start: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    found: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in found:
            continue
        found.add(name)
        stack.extend(children.get(name, ()))
    return found


def add_palm_frame(root: ET.Element, side: str) -> None:
    link_name = f"{side}_hand_palm_link"
    if any(link.get("name") == link_name for link in root.findall("link")):
        raise ValueError(f"source already defines {link_name}")
    root.append(ET.Element("link", {"name": link_name}))
    joint = ET.Element("joint", {"name": f"{side}_hand_palm_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": f"0 0 {PALM_CENTER_Z_M}", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": f"{side}_base_link"})
    ET.SubElement(joint, "child", {"link": link_name})
    root.append(joint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--unitree-commit", default=EXPECTED_UNITREE_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.unitree_commit != EXPECTED_UNITREE_COMMIT:
        raise ValueError("unreviewed Unitree asset revision")

    tree = ET.parse(args.source)
    root = tree.getroot()
    if root.get("name") != "g1_29dof_mode_15_brainco_hand":
        raise ValueError("unexpected source robot")
    source_dir = args.source.parent
    for mesh in root.findall(".//mesh"):
        path = (source_dir / mesh.get("filename", "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))

    # The upstream file gives this fixed joint the same name as its child
    # link. Pinocchio permits parsing it but cannot reduce the collision model
    # because the resulting JOINT and BODY frames are ambiguous.
    for joint in root.findall("joint"):
        if joint.get("name") == "right_thumb_tip":
            joint.set("name", "right_thumb_tip_fixed_joint")

    # Upstream represents ten mechanically fixed fingertips as zero-range
    # revolute joints (four left joints are fixed at +1 rad). Isaac Lab rejects
    # these degenerate limits. Fold the fixed angle into the origin and express
    # them as honest fixed joints without changing their physical pose.
    fixed_zero_range: list[str] = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") != "revolute" or limit is None:
            continue
        lower = float(limit.get("lower", "nan"))
        upper = float(limit.get("upper", "nan"))
        if lower != upper:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        if origin is None or axis is None:
            raise ValueError(f"malformed zero-range joint {joint.get('name')}")
        axis_values = [float(value) for value in axis.get("xyz", "").split()]
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        if lower != 0.0:
            if axis_values != [0.0, 1.0, 0.0] or rpy[0] != 0.0 or rpy[2] != 0.0:
                raise ValueError(f"unsupported fixed-angle joint {joint.get('name')}")
            rpy[1] += lower
            origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))
        joint.set("type", "fixed")
        joint.remove(axis)
        joint.remove(limit)
        fixed_zero_range.append(joint.get("name", ""))

    # Match the dark safety hand in the real demonstrations while retaining
    # the exact official Revo2 geometry and collision meshes.
    hand_links = descendants(root, "left_base_link") | descendants(root, "right_base_link")
    for link in root.findall("link"):
        if link.get("name") not in hand_links:
            continue
        for visual in link.findall("visual"):
            material = visual.find("material")
            if material is None:
                material = ET.SubElement(visual, "material")
            material.set("name", "brainco_dark")
            color = material.find("color")
            if color is None:
                color = ET.SubElement(material, "color")
            color.set("rgba", "0.025 0.028 0.032 1")

    for side in ("left", "right"):
        add_palm_frame(root, side)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    metadata = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "prepared_sha256": sha256(args.output),
        "unitree_ros_commit": args.unitree_commit,
        "semantic_palm_parent": "{side}_base_link",
        "semantic_palm_xyz_m": [0.0, 0.0, PALM_CENTER_Z_M],
        "brainco_joint_policy": "locked_open_outside_vla_contract",
        "vla_virtual_grippers": [0.0, 0.0],
        "normalized_fixed_fingertip_joints": fixed_zero_range,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"G1_BRAINCO_PREPARED urdf={args.output} sha256={metadata['prepared_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
