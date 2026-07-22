#!/usr/bin/env python3
"""Build a G1 29-DOF URDF with exact BrainCo Revo2 hand geometry.

The source G1 URDF contains Unitree Dex3 hand subtrees.  This builder removes
those subtrees, attaches the pinned BrainCo left/right URDFs at the wrist-yaw
frames, strips URDF-only debug axes, and writes absolute mesh paths so Isaac's
URDF converter can resolve every asset reproducibly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


SIDES = ("left", "right")


def _absolute_meshes(root: ET.Element, source_dir: Path) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))


def _strip_non_mesh_visuals(root: ET.Element) -> None:
    """Remove coordinate axes and colored debug-tip markers from hand URDFs."""
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            if visual.find("geometry/mesh") is None:
                link.remove(visual)


def _darken_hand(root: ET.Element) -> None:
    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        material.set("name", "brainco_dark")
        color = material.find("color")
        if color is None:
            color = ET.SubElement(material, "color")
        color.set("rgba", "0.025 0.028 0.032 1")


def _remove_dex_hands(g1: ET.Element) -> None:
    removable_links = {
        link.get("name")
        for link in g1.findall("link")
        if "_hand_" in (link.get("name") or "")
    }
    for link in list(g1.findall("link")):
        if link.get("name") in removable_links:
            g1.remove(link)
    for joint in list(g1.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        names = {
            joint.get("name") or "",
            "" if parent is None else parent.get("link", ""),
            "" if child is None else child.get("link", ""),
        }
        if any("_hand_" in name for name in names):
            g1.remove(joint)


def _attach_hand(g1: ET.Element, hand_path: Path, side: str) -> None:
    hand = ET.parse(hand_path).getroot()
    _absolute_meshes(hand, hand_path.parent)
    _strip_non_mesh_visuals(hand)
    _darken_hand(hand)

    mount_name = f"{side}_brainco_mount_link"
    for element in hand.findall("link"):
        if element.get("name") == "base_link":
            element.set("name", mount_name)
    for joint in hand.findall("joint"):
        if joint.get("name") == "base":
            joint.set("name", f"{side}_brainco_base_joint")
        for tag in ("parent", "child"):
            endpoint = joint.find(tag)
            if endpoint is not None and endpoint.get("link") == "base_link":
                endpoint.set("link", mount_name)

    mount = ET.Element("joint", {"name": f"{side}_brainco_mount_joint", "type": "fixed"})
    sign = "0.003" if side == "left" else "-0.003"
    ET.SubElement(mount, "origin", {"xyz": f"0.0415 {sign} 0", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": f"{side}_wrist_yaw_link"})
    ET.SubElement(mount, "child", {"link": mount_name})
    g1.append(mount)

    for element in hand:
        if element.tag == "mujoco":
            continue
        g1.append(deepcopy(element))


def _validate(root: ET.Element) -> None:
    links = [element.get("name") for element in root.findall("link")]
    joints = [element.get("name") for element in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError("duplicate link names in combined URDF")
    if len(joints) != len(set(joints)):
        raise ValueError("duplicate joint names in combined URDF")
    link_set = set(links)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} has no parent/child")
        if parent.get("link") not in link_set or child.get("link") not in link_set:
            raise ValueError(f"joint {joint.get('name')} refers to a missing link")
    required = {
        "left_base_link",
        "right_base_link",
        "left_pinky_tip",
        "right_pinky_tip",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    }
    missing = required - link_set
    if missing:
        raise ValueError(f"required BrainCo/G1 links missing: {sorted(missing)}")
    if any("_hand_" in name for name in links if name):
        raise ValueError("Dex hand links remain after pruning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--brainco-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    g1 = ET.parse(args.g1).getroot()
    g1.set("name", "g1_29dof_brainco_revo2")
    _absolute_meshes(g1, args.g1.parent)
    _remove_dex_hands(g1)
    for side in SIDES:
        _attach_hand(g1, args.brainco_dir / f"brainco_{side}.urdf", side)
    _validate(g1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(g1, space="  ")
    ET.ElementTree(g1).write(args.output, encoding="utf-8", xml_declaration=True)
    print(
        f"G1_BRAINCO_URDF_OK output={args.output} "
        f"links={len(g1.findall('link'))} joints={len(g1.findall('joint'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
