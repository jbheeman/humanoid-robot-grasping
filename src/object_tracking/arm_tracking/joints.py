"""Canonical G1 29-DOF arm joint contract and offline URDF checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINT_NAMES)
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

# Unitree G1 29-DOF LowState/LowCmd slots. Wrist pitch/yaw do not exist on
# 23-DOF G1 variants, so the contract must be audited against the actual robot.
LEFT_ARM_INDICES = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_INDICES = (22, 23, 24, 25, 26, 27, 28)
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
ARM_WEIGHT_INDEX = 29

# Unitree description limits, before the bridge applies its additional margin.
DEFAULT_RIGHT_JOINT_LIMITS = (
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.9722, 1.9722),
    (-1.6144, 1.6144),
    (-1.6144, 1.6144),
)


def joint_contract() -> list[dict[str, Any]]:
    """Return the right-arm command order shared by IK, HTTP, and SDK2."""

    return [
        {
            "position": position,
            "name": name,
            "sdk_index": sdk_index,
            "lower_rad": limits[0],
            "upper_rad": limits[1],
        }
        for position, (name, sdk_index, limits) in enumerate(
            zip(
                RIGHT_ARM_JOINT_NAMES,
                RIGHT_ARM_INDICES,
                DEFAULT_RIGHT_JOINT_LIMITS,
                strict=True,
            )
        )
    ]


def audit_urdf(path: str | Path, *, tolerance_rad: float = 1e-3) -> dict[str, Any]:
    """Check joint presence, order, and limits without importing Pinocchio."""

    urdf_path = Path(path)
    report: dict[str, Any] = {
        "ok": False,
        "urdf": str(urdf_path),
        "contract": joint_contract(),
        "errors": [],
        "warnings": [],
    }
    if not urdf_path.is_file():
        report["errors"].append("URDF file is missing")
        return report
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        report["errors"].append(f"URDF could not be parsed: {exc}")
        return report

    movable = [
        element for element in root.findall("joint") if element.attrib.get("type", "") != "fixed"
    ]
    movable_names = [element.attrib.get("name", "") for element in movable]
    positions: list[int] = []
    urdf_limits: dict[str, tuple[float, float]] = {}
    for name, expected_limits in zip(
        RIGHT_ARM_JOINT_NAMES, DEFAULT_RIGHT_JOINT_LIMITS, strict=True
    ):
        matches = [element for element in movable if element.attrib.get("name") == name]
        if len(matches) != 1:
            report["errors"].append(f"expected exactly one movable joint named {name!r}")
            continue
        positions.append(movable_names.index(name))
        limit = matches[0].find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            report["errors"].append(f"{name} has no finite position limits")
            continue
        try:
            actual = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        except ValueError:
            report["errors"].append(f"{name} has non-numeric position limits")
            continue
        urdf_limits[name] = actual
        if any(abs(a - b) > tolerance_rad for a, b in zip(actual, expected_limits, strict=True)):
            report["warnings"].append(
                f"{name} bridge limits {expected_limits} differ from URDF limits {actual}"
            )

    if len(positions) == 7 and positions != sorted(positions):
        report["errors"].append("right-arm URDF order does not match the HTTP/SDK command order")
    report["urdf_joint_positions"] = positions
    report["urdf_limits"] = urdf_limits
    report["ok"] = not report["errors"]
    return report
