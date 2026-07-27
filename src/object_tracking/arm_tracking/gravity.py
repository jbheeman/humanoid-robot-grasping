"""Small URDF gravity model for the robot-local 250 Hz arm loop.

The G1 image cannot load the Pinocchio wheel used by the GB10 because its
glibc is older.  This module evaluates only the static gravity term needed by
the seven right-arm joints, using the same pinned URDF and no binary runtime
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

from .joints import RIGHT_ARM_GRAVITY_FF_LIMITS_NM, RIGHT_ARM_JOINT_NAMES


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
_IDENTITY: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _vector(text: str | None, default: Vector3 = (0.0, 0.0, 0.0)) -> Vector3:
    if not text:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected a finite three-vector, got {text!r}")
    return values


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _rotate(rotation: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _rpy_rotation(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _axis_rotation(axis: Vector3, angle: float) -> Matrix3:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 0.0:
        raise ValueError("revolute joint axis must be non-zero")
    x, y, z = (value / norm for value in axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    return (
        (
            cosine + x * x * one_minus,
            x * y * one_minus - z * sine,
            x * z * one_minus + y * sine,
        ),
        (
            y * x * one_minus + z * sine,
            cosine + y * y * one_minus,
            y * z * one_minus - x * sine,
        ),
        (
            z * x * one_minus - y * sine,
            z * y * one_minus + x * sine,
            cosine + z * z * one_minus,
        ),
    )


@dataclass(frozen=True)
class _Link:
    mass_kg: float
    center_of_mass: Vector3


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    child: str
    translation: Vector3
    rotation: Matrix3
    axis: Vector3


class UrdfGravityCompensator:
    """Evaluate bounded right-arm gravity torque from a pinned URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        root_link: str = "torso_link",
        gravity_m_s2: float = 9.81,
        limits_nm: Sequence[float] = RIGHT_ARM_GRAVITY_FF_LIMITS_NM,
    ) -> None:
        path = Path(urdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"gravity URDF is missing: {path}")
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        limits = tuple(float(value) for value in limits_nm)
        if len(limits) != 7 or not all(math.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError("limits_nm must contain seven finite positive values")

        xml = ET.parse(path).getroot()
        self.links = self._parse_links(xml)
        self.children = self._parse_joints(xml)
        self.root_link = root_link
        self.gravity_m_s2 = float(gravity_m_s2)
        self.limits_nm = limits
        discovered = {
            joint.name
            for joints in self.children.values()
            for joint in joints
            if joint.name in RIGHT_ARM_JOINT_NAMES
        }
        if discovered != set(RIGHT_ARM_JOINT_NAMES):
            missing = sorted(set(RIGHT_ARM_JOINT_NAMES) - discovered)
            raise ValueError(f"gravity URDF is missing right-arm joints: {missing}")

    @staticmethod
    def _parse_links(xml: ET.Element) -> dict[str, _Link]:
        links: dict[str, _Link] = {}
        for element in xml.findall("link"):
            inertial = element.find("inertial")
            mass = 0.0
            center = (0.0, 0.0, 0.0)
            if inertial is not None:
                mass_element = inertial.find("mass")
                if mass_element is not None:
                    mass = float(mass_element.attrib.get("value", "0"))
                origin = inertial.find("origin")
                if origin is not None:
                    center = _vector(origin.attrib.get("xyz"))
            if not math.isfinite(mass) or mass < 0.0:
                raise ValueError(f"link {element.attrib.get('name')} has invalid mass")
            links[element.attrib["name"]] = _Link(mass, center)
        return links

    @staticmethod
    def _parse_joints(xml: ET.Element) -> dict[str, tuple[_Joint, ...]]:
        children: dict[str, list[_Joint]] = {}
        for element in xml.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                continue
            origin = element.find("origin")
            axis = element.find("axis")
            xyz = (0.0, 0.0, 0.0) if origin is None else _vector(origin.attrib.get("xyz"))
            rpy = (0.0, 0.0, 0.0) if origin is None else _vector(origin.attrib.get("rpy"))
            joint = _Joint(
                name=element.attrib["name"],
                kind=element.attrib.get("type", "fixed"),
                child=child_element.attrib["link"],
                translation=xyz,
                rotation=_rpy_rotation(rpy),
                axis=(1.0, 0.0, 0.0) if axis is None else _vector(axis.attrib.get("xyz")),
            )
            children.setdefault(parent_element.attrib["link"], []).append(joint)
        return {parent: tuple(joints) for parent, joints in children.items()}

    def torque(self, right_arm_q_rad: Sequence[float]) -> tuple[float, ...]:
        q = tuple(float(value) for value in right_arm_q_rad)
        if len(q) != 7 or not all(math.isfinite(value) for value in q):
            raise ValueError("right_arm_q_rad must contain seven finite positions")
        positions = dict(zip(RIGHT_ARM_JOINT_NAMES, q))
        torque = [0.0] * 7
        indices = {name: index for index, name in enumerate(RIGHT_ARM_JOINT_NAMES)}

        def visit(
            link_name: str,
            parent_rotation: Matrix3,
            parent_translation: Vector3,
            active: tuple[tuple[int, Vector3, Vector3], ...],
        ) -> None:
            link = self.links.get(link_name)
            if link is not None and link.mass_kg > 0.0:
                center = _add(parent_translation, _rotate(parent_rotation, link.center_of_mass))
                force_z = link.mass_kg * self.gravity_m_s2
                for index, origin, axis in active:
                    # Compensation is the negative generalized force produced
                    # by F=(0, 0, -m*g): axis dot ((COM-origin) x -F).
                    rx = center[0] - origin[0]
                    ry = center[1] - origin[1]
                    torque[index] += force_z * (axis[0] * ry - axis[1] * rx)

            for joint in self.children.get(link_name, ()):
                joint_rotation = _matmul(parent_rotation, joint.rotation)
                joint_origin = _add(
                    parent_translation,
                    _rotate(parent_rotation, joint.translation),
                )
                next_active = active
                angle = positions.get(joint.name, 0.0)
                if joint.kind in ("revolute", "continuous"):
                    world_axis = _rotate(joint_rotation, joint.axis)
                    if joint.name in indices:
                        next_active = (*active, (indices[joint.name], joint_origin, world_axis))
                    joint_rotation = _matmul(joint_rotation, _axis_rotation(joint.axis, angle))
                visit(joint.child, joint_rotation, joint_origin, next_active)

        visit(self.root_link, _IDENTITY, (0.0, 0.0, 0.0), ())
        return tuple(
            max(-limit, min(limit, value))
            for value, limit in zip(torque, self.limits_nm)
        )
