"""Camera deprojection, rigid transforms, planes, and pregrasp policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from math import cos, sin
from typing import Iterable, Sequence

import numpy as np


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str = "none"
    coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("intrinsic dimensions and focal lengths must be positive")
        if not np.all(np.isfinite((self.fx, self.fy, self.ppx, self.ppy))):
            raise ValueError("intrinsic values must be finite")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "fx", float(self.fx))
        object.__setattr__(self, "fy", float(self.fy))
        object.__setattr__(self, "ppx", float(self.ppx))
        object.__setattr__(self, "ppy", float(self.ppy))
        object.__setattr__(self, "coefficients", tuple(float(item) for item in self.coefficients))

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "distortion_model": self.distortion_model,
            "coefficients": list(self.coefficients),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CameraIntrinsics":
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            ppx=float(value["ppx"]),
            ppy=float(value["ppy"]),
            distortion_model=str(value.get("distortion_model", "none")),
            coefficients=tuple(float(item) for item in value.get("coefficients", ())),
        )


@dataclass(frozen=True)
class RigidTransform:
    """Transform points from a source frame into a destination frame."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = _vector3(self.translation, "translation")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation determinant must be +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_xyz_rpy(cls, xyz: Iterable[float], rpy_rad: Iterable[float]) -> "RigidTransform":
        roll, pitch, yaw = _vector3(rpy_rad, "rpy_rad")
        rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
        ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
        rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
        return cls(rz @ ry @ rx, _vector3(xyz, "xyz"))

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RigidTransform":
        return cls(value["rotation"], value["translation"])

    def to_dict(self) -> dict[str, list[object]]:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    def apply(self, point: Iterable[float] | np.ndarray) -> np.ndarray:
        points = np.asarray(point, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError("point values must be finite")
        if points.shape == (3,):
            return self.rotation @ points + self.translation
        if points.ndim == 2 and points.shape[1] == 3:
            return points @ self.rotation.T + self.translation
        raise ValueError("point must have shape (3,) or (N, 3)")

    def inverse(self) -> "RigidTransform":
        rotation = self.rotation.T
        return RigidTransform(rotation, -(rotation @ self.translation))

    def then(self, next_transform: "RigidTransform") -> "RigidTransform":
        """Apply this transform, followed by ``next_transform``."""

        return RigidTransform(
            next_transform.rotation @ self.rotation,
            next_transform.rotation @ self.translation + next_transform.translation,
        )


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float

    def __post_init__(self) -> None:
        normal = _vector3(self.normal, "normal")
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-12 or not np.isfinite(self.offset):
            raise ValueError("plane must have a finite nonzero normal and offset")
        object.__setattr__(self, "normal", normal / magnitude)
        object.__setattr__(self, "offset", float(self.offset) / magnitude)

    def signed_distance(self, points: Iterable[float] | np.ndarray) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64)
        return value @ self.normal + self.offset


_SUPPORT_EDGE_NAMES = frozenset(("u_min", "u_max", "v_min", "v_max"))


@dataclass(frozen=True)
class SupportRegion:
    """A live tabletop plane with explicitly certified physical boundaries.

    An uncertified boundary is deliberately treated as extending forever.  A
    point may therefore bypass the plane-height constraint only by crossing an
    edge whose physical location was calibrated.  For the G1 tabletop setup we
    initially certify only the robot-facing edge; that lets the hand safely
    rise from beside the hip without claiming that unseen table sides are free.
    """

    plane: Plane
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    minimum_uv: np.ndarray
    maximum_uv: np.ndarray
    certified_edges: tuple[str, ...] = ()
    edge_sources: tuple[tuple[str, str], ...] = ()
    lateral_margin_m: float = 0.07
    source: str = "unknown"

    def __post_init__(self) -> None:
        origin = _vector3(self.origin, "origin")
        axis_u = _vector3(self.axis_u, "axis_u")
        axis_v = _vector3(self.axis_v, "axis_v")
        minimum_uv = np.asarray(self.minimum_uv, dtype=np.float64)
        maximum_uv = np.asarray(self.maximum_uv, dtype=np.float64)
        if minimum_uv.shape != (2,) or maximum_uv.shape != (2,):
            raise ValueError("support bounds must contain two values")
        if not np.all(np.isfinite(np.concatenate((minimum_uv, maximum_uv)))):
            raise ValueError("support bounds must be finite")
        if np.any(maximum_uv <= minimum_uv):
            raise ValueError("support maximum must exceed minimum")
        if not np.isfinite(self.lateral_margin_m) or self.lateral_margin_m < 0.0:
            raise ValueError("support lateral margin must be finite and non-negative")
        if float(np.dot(self.plane.normal, (0.0, 0.0, 1.0))) < np.cos(np.deg2rad(15.0)):
            raise ValueError("support plane is not a tabletop-like upward plane")
        if not np.isclose(np.linalg.norm(axis_u), 1.0, atol=1e-6) or not np.isclose(
            np.linalg.norm(axis_v), 1.0, atol=1e-6
        ):
            raise ValueError("support axes must be unit length")
        if (
            abs(float(np.dot(axis_u, axis_v))) > 1e-6
            or abs(float(np.dot(axis_u, self.plane.normal))) > 1e-6
            or abs(float(np.dot(axis_v, self.plane.normal))) > 1e-6
        ):
            raise ValueError("support axes must form an orthogonal plane basis")
        certified = tuple(dict.fromkeys(str(edge) for edge in self.certified_edges))
        if any(edge not in _SUPPORT_EDGE_NAMES for edge in certified):
            raise ValueError("support region contains an unknown certified edge")
        edge_sources = tuple((str(edge), str(source)) for edge, source in self.edge_sources)
        if any(edge not in _SUPPORT_EDGE_NAMES for edge, _ in edge_sources):
            raise ValueError("support region edge provenance contains an unknown edge")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis_u", axis_u)
        object.__setattr__(self, "axis_v", axis_v)
        object.__setattr__(self, "minimum_uv", minimum_uv)
        object.__setattr__(self, "maximum_uv", maximum_uv)
        object.__setattr__(self, "certified_edges", certified)
        object.__setattr__(self, "edge_sources", edge_sources)
        object.__setattr__(self, "lateral_margin_m", float(self.lateral_margin_m))
        object.__setattr__(self, "source", str(self.source))

    @classmethod
    def from_xy_bounds(
        cls,
        plane: Plane,
        minimum_xy: Iterable[float],
        maximum_xy: Iterable[float],
        *,
        certified_edges: tuple[str, ...] = ("u_min",),
        lateral_margin_m: float = 0.07,
        source: str = "calibrated_workspace",
    ) -> "SupportRegion":
        minimum = np.asarray(tuple(minimum_xy), dtype=np.float64)
        maximum = np.asarray(tuple(maximum_xy), dtype=np.float64)
        if minimum.shape != (2,) or maximum.shape != (2,):
            raise ValueError("XY support bounds must contain two values")
        if float(np.dot(plane.normal, (0.0, 0.0, 1.0))) < np.cos(np.deg2rad(15.0)):
            raise ValueError("support plane is not a tabletop-like upward plane")
        axis_u = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        axis_u -= float(np.dot(axis_u, plane.normal)) * plane.normal
        magnitude = float(np.linalg.norm(axis_u))
        if magnitude <= 1e-9:
            raise ValueError("support plane cannot define a forward tangent axis")
        axis_u /= magnitude
        axis_v = np.cross(plane.normal, axis_u)
        axis_v /= np.linalg.norm(axis_v)
        origin = -plane.offset * plane.normal
        corners = []
        for x in (minimum[0], maximum[0]):
            for y in (minimum[1], maximum[1]):
                z = -(plane.offset + plane.normal[0] * x + plane.normal[1] * y) / plane.normal[2]
                corners.append((x, y, z))
        relative = np.asarray(corners, dtype=np.float64) - origin
        uv = np.column_stack((relative @ axis_u, relative @ axis_v))
        return cls(
            plane=plane,
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
            minimum_uv=uv.min(axis=0),
            maximum_uv=uv.max(axis=0),
            certified_edges=certified_edges,
            edge_sources=tuple(
                (edge, "calibrated") for edge in certified_edges
            ),
            lateral_margin_m=lateral_margin_m,
            source=source,
        )

    @classmethod
    def from_ordered_corners(
        cls,
        plane: Plane,
        corners_xyz: Sequence[Iterable[float]],
        *,
        certified_edges: tuple[str, ...] = ("u_min", "u_max", "v_min", "v_max"),
        lateral_margin_m: float = 0.07,
        source: str = "calibrated_tabletop_corners",
    ) -> "SupportRegion":
        """Build a tabletop region from near-left/right then far-right/left corners."""

        corners = np.asarray([_vector3(point, "corner") for point in corners_xyz])
        if corners.shape != (4, 3):
            raise ValueError("support region requires four ordered corners")
        if np.max(np.abs(plane.signed_distance(corners))) > 0.01:
            raise ValueError("support corners are not on the live tabletop plane")
        near_midpoint = 0.5 * (corners[0] + corners[1])
        far_midpoint = 0.5 * (corners[2] + corners[3])
        axis_v = corners[1] - corners[0]
        axis_v -= float(np.dot(axis_v, plane.normal)) * plane.normal
        axis_v_norm = float(np.linalg.norm(axis_v))
        if axis_v_norm <= 0.05:
            raise ValueError("tabletop near edge is too short")
        axis_v /= axis_v_norm
        axis_u = far_midpoint - near_midpoint
        axis_u -= float(np.dot(axis_u, plane.normal)) * plane.normal
        axis_u -= float(np.dot(axis_u, axis_v)) * axis_v
        axis_u_norm = float(np.linalg.norm(axis_u))
        if axis_u_norm <= 0.05:
            raise ValueError("tabletop depth is too short")
        axis_u /= axis_u_norm
        origin = near_midpoint - float(plane.signed_distance(near_midpoint)) * plane.normal
        relative = corners - origin
        uv = np.column_stack((relative @ axis_u, relative @ axis_v))
        minimum_uv = uv.min(axis=0)
        maximum_uv = uv.max(axis=0)
        if np.any(maximum_uv - minimum_uv < 0.05):
            raise ValueError("tabletop footprint is degenerate")
        return cls(
            plane=plane,
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
            minimum_uv=minimum_uv,
            maximum_uv=maximum_uv,
            certified_edges=certified_edges,
            edge_sources=tuple(
                (edge, "calibrated") for edge in certified_edges
            ),
            lateral_margin_m=lateral_margin_m,
            source=source,
        )

    def signed_distance(self, points: Iterable[float] | np.ndarray) -> np.ndarray:
        return self.plane.signed_distance(points)

    def projection_uv(self, point: Iterable[float]) -> np.ndarray:
        relative = _vector3(point, "point") - self.origin
        return np.asarray(
            (float(np.dot(relative, self.axis_u)), float(np.dot(relative, self.axis_v))),
            dtype=np.float64,
        )

    def requires_clearance(self, point: Iterable[float]) -> bool:
        u, v = self.projection_uv(point)
        margin = self.lateral_margin_m
        outside = {
            "u_min": u < self.minimum_uv[0] - margin,
            "u_max": u > self.maximum_uv[0] + margin,
            "v_min": v < self.minimum_uv[1] - margin,
            "v_max": v > self.maximum_uv[1] + margin,
        }
        return not any(outside[edge] for edge in self.certified_edges)

    def edge_source(self, edge: str) -> str:
        if edge not in _SUPPORT_EDGE_NAMES:
            raise ValueError(f"unknown support edge: {edge}")
        return dict(self.edge_sources).get(
            edge,
            "observed" if edge in self.certified_edges else "unknown",
        )

    def classify_point(
        self,
        point: Iterable[float],
        *,
        side_margin_m: float = 0.0,
        top_clearance_m: float = 0.05,
    ) -> str:
        """Classify a point relative to the conservatively inflated table prism."""

        if side_margin_m < 0.0 or top_clearance_m < 0.0:
            raise ValueError("table prism margins must be non-negative")
        u, v = self.projection_uv(point)
        inside = bool(
            self.minimum_uv[0] - side_margin_m <= u <= self.maximum_uv[0] + side_margin_m
            and self.minimum_uv[1] - side_margin_m <= v <= self.maximum_uv[1] + side_margin_m
        )
        height = float(self.signed_distance(point))
        if inside and height < top_clearance_m:
            return "under_or_inside"
        if height >= top_clearance_m:
            return "above_clearance"
        return "outside"

    def has_clearance(self, point: Iterable[float], *, minimum_clearance_m: float) -> bool:
        if minimum_clearance_m < 0.0 or not np.isfinite(minimum_clearance_m):
            raise ValueError("minimum clearance must be finite and non-negative")
        if not self.requires_clearance(point):
            return True
        return bool(float(self.signed_distance(point)) >= minimum_clearance_m)

    def project_to_clearance(
        self,
        point: Iterable[float],
        *,
        minimum_clearance_m: float,
    ) -> np.ndarray:
        """Project a point onto the tabletop's safe half-space.

        The projection is the minimum Euclidean correction for a plane: move
        only along the plane normal.  It applies only over the certified table
        footprint.  Full-link and swept-path clearance remain the IK layer's
        responsibility; this projection keeps a low VLA end-effector proposal
        useful instead of immediately rejecting it.
        """

        value = _vector3(point, "point")
        if minimum_clearance_m < 0.0 or not np.isfinite(minimum_clearance_m):
            raise ValueError("minimum clearance must be finite and non-negative")
        if not self.requires_clearance(value):
            return value.copy()
        distance = float(self.signed_distance(value))
        correction = max(0.0, minimum_clearance_m - distance)
        return value + correction * self.plane.normal

    def to_dict(self) -> dict[str, object]:
        return {
            "normal": self.plane.normal.tolist(),
            "offset": float(self.plane.offset),
            "footprint": {
                "origin": self.origin.tolist(),
                "axis_u": self.axis_u.tolist(),
                "axis_v": self.axis_v.tolist(),
                "minimum_uv": self.minimum_uv.tolist(),
                "maximum_uv": self.maximum_uv.tolist(),
                "certified_edges": list(self.certified_edges),
                "edge_sources": dict(self.edge_sources),
                "lateral_margin_m": self.lateral_margin_m,
                "source": self.source,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SupportRegion":
        footprint = value.get("footprint")
        if not isinstance(footprint, dict):
            raise ValueError("support region has no bounded footprint")
        return cls(
            plane=Plane(value["normal"], float(value["offset"])),
            origin=footprint["origin"],
            axis_u=footprint["axis_u"],
            axis_v=footprint["axis_v"],
            minimum_uv=footprint["minimum_uv"],
            maximum_uv=footprint["maximum_uv"],
            certified_edges=tuple(footprint.get("certified_edges", ())),
            edge_sources=tuple(
                (str(edge), str(source))
                for edge, source in dict(footprint.get("edge_sources", {})).items()
            ),
            lateral_margin_m=float(footprint.get("lateral_margin_m", 0.07)),
            source=str(footprint.get("source", "unknown")),
        )


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = _vector3(self.minimum, "minimum")
        maximum = _vector3(self.maximum, "maximum")
        if np.any(maximum <= minimum):
            raise ValueError("workspace maximum must exceed minimum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, point: Iterable[float], margin_m: float = 0.0) -> bool:
        value = _vector3(point, "point")
        if margin_m < 0:
            raise ValueError("margin_m must be non-negative")
        return bool(
            np.all(value >= self.minimum + margin_m) and np.all(value <= self.maximum - margin_m)
        )


@dataclass(frozen=True)
class TargetPose:
    position: np.ndarray
    orientation_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = _vector3(self.position, "position")
        orientation = np.asarray(self.orientation_xyzw, dtype=np.float64)
        norm = float(np.linalg.norm(orientation))
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)) or norm <= 1e-12:
            raise ValueError("orientation_xyzw must be a finite quaternion")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation_xyzw", orientation / norm)


def map_pixel_between_profiles(
    pixel_xy: tuple[float, float],
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> tuple[float, float]:
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    if min(source_width, source_height, destination_width, destination_height) <= 0:
        raise ValueError("profile dimensions must be positive")
    return (
        pixel_xy[0] * destination_width / source_width,
        pixel_xy[1] * destination_height / source_height,
    )


def deproject_pixel(
    pixel_xy: tuple[float, float], depth_m: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Deproject a registered pixel using RealSense Brown distortion conventions."""

    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be positive and finite")
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("pixel_xy must contain two finite values")
    x = (pixel[0] - intrinsics.ppx) / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) / intrinsics.fy
    x, y = _undistort_normalized(x, y, intrinsics)
    return np.array([x * depth_m, y * depth_m, depth_m], dtype=np.float64)


def intersect_pixel_ray_with_plane(
    pixel_xy: Iterable[float],
    intrinsics: CameraIntrinsics,
    optical_to_base: RigidTransform,
    plane: Plane,
) -> np.ndarray:
    """Intersect a calibrated camera pixel ray with a plane in base coordinates."""

    pixel = np.asarray(tuple(pixel_xy), dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("pixel_xy must contain two finite values")
    x = (pixel[0] - intrinsics.ppx) / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) / intrinsics.fy
    x, y = _undistort_normalized(x, y, intrinsics)
    ray = optical_to_base.rotation @ np.asarray((x, y, 1.0), dtype=np.float64)
    origin = optical_to_base.translation
    denominator = float(np.dot(plane.normal, ray))
    if abs(denominator) <= 1e-9:
        raise ValueError("camera ray is parallel to plane")
    distance = -float(np.dot(plane.normal, origin) + plane.offset) / denominator
    if distance <= 0.0 or not np.isfinite(distance):
        raise ValueError("camera ray intersects plane behind camera")
    return origin + distance * ray


def _undistort_normalized(
    x: np.ndarray | float, y: np.ndarray | float, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray | float, np.ndarray | float]:
    model = intrinsics.distortion_model.lower()
    coefficients = (*intrinsics.coefficients, 0.0, 0.0, 0.0, 0.0, 0.0)[:5]
    k1, k2, p1, p2, k3 = coefficients
    if model in {"none", "pinhole"}:
        pass
    elif model in {"brown_conrady", "modified_brown_conrady"}:
        distorted_x, distorted_y = x, y
        for _ in range(10):
            radius_squared = x * x + y * y
            radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
            if np.any(np.abs(radial) <= 1e-12):
                raise ValueError("distortion coefficients produce a singular projection")
            delta_x = 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)
            delta_y = p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (distorted_x - delta_x) / radial
            y = (distorted_y - delta_y) / radial
    elif model == "inverse_brown_conrady":
        radius_squared = x * x + y * y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
        x, y = (
            x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x),
            y * radial + 2.0 * p2 * x * y + p1 * (radius_squared + 2.0 * y * y),
        )
    else:
        raise ValueError(f"unsupported distortion model {intrinsics.distortion_model!r}")
    return x, y


def fit_plane(points: np.ndarray, *, orient_toward: Iterable[float] | None = None) -> Plane:
    """Fit a least-squares plane to already filtered 3D support points."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("at least three 3D points are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("plane points must be finite")
    centroid = values.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(values - centroid, full_matrices=False)
    if singular_values[-2] <= 1e-9:
        raise ValueError("plane points are collinear")
    normal = vh[-1]
    if orient_toward is not None and np.dot(normal, _vector3(orient_toward, "orient_toward")) < 0:
        normal = -normal
    return Plane(normal=normal, offset=-float(np.dot(normal, centroid)))


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float = 0.015,
    iterations: int = 100,
    minimum_inlier_fraction: float = 0.35,
    orient_toward: Iterable[float] | None = None,
    seed: int = 0,
) -> tuple[Plane, np.ndarray]:
    """Extract the dominant plane and return its input-aligned inlier mask."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("at least three 3D points are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("plane points must be finite")
    if distance_threshold_m <= 0 or iterations <= 0 or not 0 < minimum_inlier_fraction <= 1:
        raise ValueError("invalid RANSAC thresholds")
    generator = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = values[generator.choice(len(values), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-9:
            continue
        normal /= magnitude
        offset = -float(np.dot(normal, sample[0]))
        mask = np.abs(values @ normal + offset) <= distance_threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    required = max(3, int(np.ceil(len(values) * minimum_inlier_fraction)))
    if best_mask is None or best_count < required:
        raise ValueError("no dominant support plane met the inlier requirement")
    return fit_plane(values[best_mask], orient_toward=orient_toward), best_mask


@dataclass(frozen=True)
class SupportDetectionDiagnostics:
    candidate_count: int
    inlier_count: int
    connected_inlier_count: int
    inlier_fraction: float
    normal_tilt_deg: float
    residual_median_m: float
    residual_p95_m: float
    observed_size_m: tuple[float, float]
    expected_size_m: tuple[float, float] | None
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "inlier_count": self.inlier_count,
            "connected_inlier_count": self.connected_inlier_count,
            "inlier_fraction": round(self.inlier_fraction, 5),
            "normal_tilt_deg": round(self.normal_tilt_deg, 3),
            "residual_median_mm": round(self.residual_median_m * 1000.0, 3),
            "residual_p95_mm": round(self.residual_p95_m * 1000.0, 3),
            "observed_size_m": [round(value, 4) for value in self.observed_size_m],
            "expected_size_m": (
                None
                if self.expected_size_m is None
                else [round(value, 4) for value in self.expected_size_m]
            ),
            "score": round(self.score, 5),
        }


def _largest_connected_sample_mask(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    stride: int,
) -> np.ndarray:
    """Select one 4-connected component from sparse organized depth samples."""

    if len(rows) != len(columns):
        raise ValueError("sample rows and columns must have equal length")
    locations = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(zip(rows, columns))
    }
    remaining = set(locations)
    largest: list[int] = []
    while remaining:
        start = remaining.pop()
        queue = deque((start,))
        component = [locations[start]]
        while queue:
            row, column = queue.popleft()
            for neighbor in (
                (row - stride, column),
                (row + stride, column),
                (row, column - stride),
                (row, column + stride),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(locations[neighbor])
        if len(component) > len(largest):
            largest = component
    mask = np.zeros(len(rows), dtype=bool)
    mask[largest] = True
    return mask


def detect_automatic_support_region(
    z16: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
    optical_to_base: RigidTransform,
    base_minimum: Iterable[float],
    base_maximum: Iterable[float],
    expected_size_m: tuple[float, float] | None = None,
    stride: int = 6,
    distance_threshold_m: float = 0.012,
    minimum_connected_inliers: int = 80,
) -> tuple[SupportRegion, SupportDetectionDiagnostics]:
    """Detect a conservative tabletop pose and footprint from registered RGB-D.

    Live depth supplies the plane pose and visible near edge. Saved physical
    dimensions, when available, conservatively complete boundaries that may be
    hidden by the plush or arm.
    """

    points, rows, columns = deproject_depth_samples_with_sample_grid(
        z16,
        intrinsics,
        depth_scale=depth_scale,
        stride=stride,
    )
    points = optical_to_base.apply(points)
    minimum = _vector3(base_minimum, "base_minimum")
    maximum = _vector3(base_maximum, "base_maximum")
    bounded = np.all((points >= minimum) & (points <= maximum), axis=1)
    points, rows, columns = points[bounded], rows[bounded], columns[bounded]
    if len(points) < minimum_connected_inliers:
        raise ValueError("too few tabletop-height depth samples")

    remaining = np.ones(len(points), dtype=bool)
    candidates: list[tuple[float, Plane, np.ndarray, np.ndarray, SupportDetectionDiagnostics]] = []
    for candidate_index in range(3):
        subset_indices = np.flatnonzero(remaining)
        if len(subset_indices) < minimum_connected_inliers:
            break
        try:
            plane, local_inliers = fit_plane_ransac(
                points[subset_indices],
                distance_threshold_m=distance_threshold_m,
                iterations=140,
                minimum_inlier_fraction=0.12,
                orient_toward=(0.0, 0.0, 1.0),
                seed=candidate_index,
            )
        except ValueError:
            break
        inlier_indices = subset_indices[local_inliers]
        tilt_deg = math.degrees(
            math.acos(float(np.clip(np.dot(plane.normal, (0.0, 0.0, 1.0)), -1.0, 1.0)))
        )
        remaining[inlier_indices] = False
        if tilt_deg > 15.0:
            continue
        connected_local = _largest_connected_sample_mask(
            rows[inlier_indices],
            columns[inlier_indices],
            stride=stride,
        )
        connected_indices = inlier_indices[connected_local]
        if len(connected_indices) < minimum_connected_inliers:
            continue
        connected = points[connected_indices]
        axis_u = np.asarray((1.0, 0.0, 0.0), dtype=float)
        axis_u -= float(np.dot(axis_u, plane.normal)) * plane.normal
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(plane.normal, axis_u)
        axis_v /= np.linalg.norm(axis_v)
        origin = -plane.offset * plane.normal
        relative = connected - origin
        uv = np.column_stack((relative @ axis_u, relative @ axis_v))
        low = np.percentile(uv, 2.0, axis=0)
        high = np.percentile(uv, 98.0, axis=0)
        observed_size = high - low
        if np.any(observed_size < 0.12):
            continue
        residual = np.abs(plane.signed_distance(connected))
        inlier_fraction = len(inlier_indices) / len(points)
        size_agreement = 1.0
        if expected_size_m is not None:
            expected = np.asarray(expected_size_m, dtype=float)
            size_agreement = float(
                np.exp(-np.sum(np.abs(np.log(np.maximum(observed_size, 1e-3) / expected))))
            )
        score = (
            2.0 * inlier_fraction
            + len(connected_indices) / len(points)
            + size_agreement
            - tilt_deg / 15.0
            - float(np.percentile(residual, 95.0)) / distance_threshold_m
        )
        diagnostics = SupportDetectionDiagnostics(
            candidate_count=0,
            inlier_count=len(inlier_indices),
            connected_inlier_count=len(connected_indices),
            inlier_fraction=inlier_fraction,
            normal_tilt_deg=tilt_deg,
            residual_median_m=float(np.median(residual)),
            residual_p95_m=float(np.percentile(residual, 95.0)),
            observed_size_m=(float(observed_size[0]), float(observed_size[1])),
            expected_size_m=expected_size_m,
            score=score,
        )
        candidates.append((score, plane, low, high, diagnostics))
    if not candidates:
        raise ValueError("no connected horizontal tabletop candidate")
    _, plane, low, high, diagnostics = max(candidates, key=lambda item: item[0])
    if expected_size_m is not None:
        expected = np.asarray(expected_size_m, dtype=float)
        # The robot-facing depth edge is normally visible. Complete the far
        # and lateral boundaries from saved dimensions so occlusion cannot
        # make the table look smaller and therefore less restrictive.
        minimum_uv = np.asarray((low[0], 0.5 * (low[1] + high[1] - expected[1])))
        maximum_uv = minimum_uv + expected
    else:
        minimum_uv, maximum_uv = low, high
    axis_u = np.asarray((1.0, 0.0, 0.0), dtype=float)
    axis_u -= float(np.dot(axis_u, plane.normal)) * plane.normal
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(plane.normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    support = SupportRegion(
        plane=plane,
        origin=-plane.offset * plane.normal,
        axis_u=axis_u,
        axis_v=axis_v,
        minimum_uv=minimum_uv,
        maximum_uv=maximum_uv,
        certified_edges=("u_min",),
        edge_sources=(
            ("u_min", "observed"),
            ("u_max", "prior_estimated"),
            ("v_min", "prior_estimated"),
            ("v_max", "prior_estimated"),
        ),
        lateral_margin_m=0.07,
        source="automatic_rgbd_plane_dimension_prior",
    )
    return support, SupportDetectionDiagnostics(
        **{
            **diagnostics.__dict__,
            "candidate_count": len(candidates),
        }
    )


def deproject_depth_samples(
    z16: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
    stride: int = 4,
    minimum_depth_m: float = 0.12,
    maximum_depth_m: float = 4.0,
    row_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert a bounded, strided registered depth region into optical-frame points."""

    depth = np.asarray(z16)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth frame shape must match its calibrated intrinsics")
    if stride <= 0 or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("stride and depth scale must be positive")
    start, stop = row_range or (0, intrinsics.height)
    if not 0 <= start < stop <= intrinsics.height:
        raise ValueError("row_range is outside the depth frame")
    rows, columns = np.mgrid[start:stop:stride, 0 : intrinsics.width : stride]
    depths = depth[rows, columns].astype(np.float64) * depth_scale
    valid = np.isfinite(depths) & (depths >= minimum_depth_m) & (depths <= maximum_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    pixels_x = columns[valid].astype(np.float64)
    pixels_y = rows[valid].astype(np.float64)
    depths = depths[valid]
    x = (pixels_x - intrinsics.ppx) / intrinsics.fx
    y = (pixels_y - intrinsics.ppy) / intrinsics.fy
    x, y = _undistort_normalized(x, y, intrinsics)
    return np.column_stack((x * depths, y * depths, depths))


def _points_in_polygon(points_xy: np.ndarray, polygon_uv: Sequence[tuple[float, float]]) -> np.ndarray:
    """Return a boolean mask for points inside or on the boundary of a polygon."""
    polygon = np.asarray(polygon_uv, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must contain at least three (u, v) corners")
    if points_xy.shape[1] != 2:
        raise ValueError("points_xy must be (N, 2)")
    px = points_xy[:, 0]
    py = points_xy[:, 1]
    x = polygon[:, 0]
    y = polygon[:, 1]

    inside = np.zeros(px.shape, dtype=bool)
    for i in range(len(polygon)):
        j = i - 1
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        denominator = yj - yi
        if abs(float(denominator)) <= 1e-12:
            continue
        mask = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / denominator + xi
        )
        inside ^= mask
    return inside


def extract_support_plane(
    z16: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
    optical_to_base: RigidTransform | None = None,
    lower_image_fraction: float = 0.45,
    stride: int = 4,
    pixel_roi: Sequence[tuple[float, float]] | None = None,
    base_minimum: Iterable[float] | None = None,
    base_maximum: Iterable[float] | None = None,
    distance_threshold_m: float = 0.015,
    minimum_inlier_fraction: float = 0.35,
) -> tuple[Plane, np.ndarray]:
    """Extract a dominant support plane from a depth sample region.

    By default this uses the lower image strip. If ``pixel_roi`` is provided, the
    support plane is extracted only from points inside that polygon footprint.
    """

    if not 0 < lower_image_fraction <= 1:
        raise ValueError("lower_image_fraction must be in (0, 1]")
    row_range = None if pixel_roi is not None else (int(intrinsics.height * (1.0 - lower_image_fraction)), intrinsics.height)

    points, all_rows, all_cols = deproject_depth_samples_with_sample_grid(
        z16,
        intrinsics,
        depth_scale=depth_scale,
        stride=stride,
        row_range=row_range,
    )
    if pixel_roi is not None:
        if len(pixel_roi) < 3:
            raise ValueError("pixel_roi must contain at least three points")
        sample_points = np.column_stack((all_cols.ravel(), all_rows.ravel()))
        inside = _points_in_polygon(sample_points, pixel_roi)
        points = points[inside]
    if points.size == 0:
        raise ValueError("no points available for support plane extraction")
    if (base_minimum is None) != (base_maximum is None):
        raise ValueError("base_minimum and base_maximum must be provided together")
    if optical_to_base is not None:
        points = optical_to_base.apply(points)
        orientation = (0.0, 0.0, 1.0)
    else:
        orientation = None
    if base_minimum is not None and base_maximum is not None:
        minimum = _vector3(base_minimum, "base_minimum")
        maximum = _vector3(base_maximum, "base_maximum")
        if np.any(maximum <= minimum):
            raise ValueError("base support-plane bounds are invalid")
        points = points[np.all((points >= minimum) & (points <= maximum), axis=1)]
        if len(points) < 3:
            raise ValueError("no tabletop-height points remain inside the calibrated bounds")
    return fit_plane_ransac(
        points,
        distance_threshold_m=distance_threshold_m,
        minimum_inlier_fraction=minimum_inlier_fraction,
        orient_toward=orientation,
    )


def deproject_depth_samples_with_sample_grid(
    z16: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
    row_range: tuple[int, int] | None = None,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`deproject_depth_samples` but also returns row/col sample grids."""
    depth = np.asarray(z16)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth frame shape must match its calibrated intrinsics")
    if stride <= 0 or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("stride and depth scale must be positive")
    start, stop = row_range or (0, intrinsics.height)
    if not 0 <= start < stop <= intrinsics.height:
        raise ValueError("row_range is outside the depth frame")
    rows, columns = np.mgrid[start:stop:stride, 0 : intrinsics.width : stride]
    depths = depth[rows, columns].astype(np.float64) * depth_scale
    valid = np.isfinite(depths) & (depths >= 1e-3) & (depths <= 3.0)
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float64),
            rows,
            columns,
        )
    depths = depths[valid]
    rows = rows[valid]
    columns = columns[valid]
    pixels_x = columns.astype(np.float64)
    pixels_y = rows.astype(np.float64)
    x = (pixels_x - intrinsics.ppx) / intrinsics.fx
    y = (pixels_y - intrinsics.ppy) / intrinsics.fy
    x, y = _undistort_normalized(x, y, intrinsics)
    return np.column_stack((x * depths, y * depths, depths)), rows, columns


def generate_pregrasp_target(
    object_position: Iterable[float],
    shoulder_position: Iterable[float],
    *,
    stand_off_m: float = 0.20,
    neutral_orientation_xyzw: Iterable[float] = (0.0, 0.0, 0.0, 1.0),
) -> TargetPose:
    object_point = _vector3(object_position, "object_position")
    shoulder = _vector3(shoulder_position, "shoulder_position")
    approach = object_point - shoulder
    distance = float(np.linalg.norm(approach))
    if stand_off_m <= 0 or distance <= stand_off_m:
        raise ValueError("object must be farther from the shoulder than the stand-off")
    position = object_point - approach / distance * stand_off_m
    return TargetPose(position, np.asarray(tuple(neutral_orientation_xyzw), dtype=np.float64))


def has_plane_clearance(
    point: Iterable[float], plane: Plane, *, minimum_clearance_m: float = 0.10
) -> bool:
    if minimum_clearance_m < 0:
        raise ValueError("minimum_clearance_m must be non-negative")
    return bool(plane.signed_distance(_vector3(point, "point")) >= minimum_clearance_m)


def has_support_clearance(
    point: Iterable[float],
    support: Plane | SupportRegion,
    *,
    minimum_clearance_m: float = 0.10,
) -> bool:
    if isinstance(support, SupportRegion):
        return support.has_clearance(point, minimum_clearance_m=minimum_clearance_m)
    return has_plane_clearance(point, support, minimum_clearance_m=minimum_clearance_m)
