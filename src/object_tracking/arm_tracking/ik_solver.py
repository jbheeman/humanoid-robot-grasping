from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from .joints import RIGHT_ARM_JOINT_NAMES


XR_TELEOPERATE_REVISION = "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
UNITREE_ROS_REVISION = "d96d8f63ae17a7108d4f7229c00ef875ba7129c9"
RIGHT_ARM_JOINTS = RIGHT_ARM_JOINT_NAMES
_MAX_START_ESCAPE_PENETRATION_M = 0.005
_MAX_STREAM_JOINT_STEP_RAD = 0.04

# These collision meshes overlap at the G1's factory shoulder articulation
# range.  They are kinematic neighbours, not an arm-through-torso route.  The
# remaining mesh pairs (hand/hip, forearm/torso, table clearance, joint limits)
# stay active in every planning query.
_ADJACENT_G1_COLLISION_PAIRS = {
    frozenset(("torso_link_0", "right_shoulder_pitch_link_0")),
    frozenset(("torso_link_0", "right_shoulder_roll_link_0")),
    frozenset(("torso_link_0", "right_shoulder_yaw_link_0")),
    frozenset(("torso_link_0", "right_elbow_link_0")),
    frozenset(("right_shoulder_yaw_link_0", "right_elbow_link_0")),
    frozenset(("right_elbow_link_0", "right_wrist_roll_link_0")),
}

# Deterministic local directions discovered from the measured G1 hip-rest
# posture. They are hypotheses only: _guided_start_escape densely validates
# every application against the current pose, full collision model, bounded
# support region, 5 mm penetration ceiling, collision-free latch, and 10 mm exit gate.
_G1_HIP_ESCAPE_DELTAS = (
    # The stock hip-rest pose needs a small simultaneous shoulder-pitch
    # change to clear the thumb mesh by the required 10 mm.  The full edge is
    # still densely checked below before any interpolated waypoint is used.
    (-0.10, -0.28, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-0.10, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.03, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, -0.03, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.0, 0.03),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.0, -0.03),
    (0.0, -0.20, 0.0, 0.0, 0.03, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, -0.03, 0.0, 0.0),
    (0.03, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-0.03, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.03, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, -0.03, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.03, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, -0.03, 0.0, 0.0, 0.0),
    (0.08, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-0.08, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.08, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, -0.08, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.08, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, -0.08, 0.0, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.08, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, -0.08, 0.0, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.08, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, -0.08, 0.0),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.0, 0.08),
    (0.0, -0.20, 0.0, 0.0, 0.0, 0.0, -0.08),
    (0.0, -0.24, 0.0, 0.0, 0.0, 0.12, 0.0),
    (0.0, -0.24, 0.0, 0.0, 0.0, -0.12, 0.0),
    (0.0, -0.24, 0.0, 0.0, 0.0, 0.0, 0.12),
    (0.0, -0.24, 0.0, 0.0, 0.0, 0.0, -0.12),
)


class IKUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class IKResult:
    ok: bool
    q_rad: tuple[float, ...] | None
    position_error_m: float
    orientation_error_rad: float
    reason: str | None = None


@dataclass(frozen=True)
class IKPathResult:
    ok: bool
    q_path: tuple[tuple[float, ...], ...] | None
    position_error_m: float
    orientation_error_rad: float
    reason: str | None = None


def collision_aware_joint_path(
    start_q: Sequence[float],
    goal_q: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    state_is_valid: Callable[[np.ndarray], bool],
    *,
    edge_step_rad: float = 0.04,
    extension_step_rad: float = 0.18,
    max_iterations: int = 500,
    planning_timeout_s: float = 1.5,
    seed: int = 7,
    guided_sampling: bool = True,
) -> tuple[tuple[float, ...], ...] | None:
    """Plan a deterministic bidirectional RRT path and shortcut it.

    This operates entirely on a model. Callers provide the collision predicate;
    it never communicates with hardware.
    """

    start = np.asarray(start_q, dtype=float)
    goal = np.asarray(goal_q, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if (
        start.shape != goal.shape
        or start.shape != low.shape
        or start.shape != high.shape
        or start.ndim != 1
        or not np.all(np.isfinite(np.concatenate((start, goal, low, high))))
        or np.any(low > high)
        or edge_step_rad <= 0.0
        or extension_step_rad <= 0.0
        or max_iterations <= 0
        or not np.isfinite(planning_timeout_s)
        or planning_timeout_s <= 0.0
    ):
        raise ValueError("invalid collision-aware joint path inputs")
    deadline = time.monotonic() + planning_timeout_s

    def edge_is_valid(begin: np.ndarray, end: np.ndarray) -> bool:
        steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / edge_step_rad)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            if time.monotonic() >= deadline:
                return False
            if not state_is_valid((1.0 - alpha) * begin + alpha * end):
                return False
        return True

    if not state_is_valid(start) or not state_is_valid(goal):
        return None
    if edge_is_valid(start, goal):
        return (tuple(start.tolist()), tuple(goal.tolist()))

    rng = np.random.default_rng(seed)
    nodes_a = [start]
    parents_a = [-1]
    nodes_b = [goal]
    parents_b = [-1]
    swapped = False

    def nearest(nodes: list[np.ndarray], target: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(node - target) for node in nodes]))

    def extend(nodes: list[np.ndarray], parents: list[int], target: np.ndarray) -> int | None:
        parent = nearest(nodes, target)
        begin = nodes[parent]
        delta = target - begin
        distance = float(np.linalg.norm(delta))
        end = (
            target
            if distance <= extension_step_rad
            else begin + delta / distance * extension_step_rad
        )
        # Standard RRT-Connect advances to the last valid interpolation point.
        # Rejecting an entire extension because only its tail collides prevents
        # either tree from reaching the boundary of a narrow passage.
        steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / edge_step_rad)))
        last_valid = begin
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            if time.monotonic() >= deadline:
                break
            candidate = (1.0 - alpha) * begin + alpha * end
            if not state_is_valid(candidate):
                break
            last_valid = candidate
        if np.linalg.norm(last_valid - begin) < 1e-9:
            return None
        nodes.append(last_valid)
        parents.append(parent)
        return len(nodes) - 1

    def chain(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return list(reversed(result))

    path: list[np.ndarray] | None = None
    for _ in range(max_iterations):
        if time.monotonic() >= deadline:
            break
        choice = rng.random()
        if choice < 0.15:
            sample = goal
        elif guided_sampling and choice < 0.70:
            alpha = rng.random()
            center = (1.0 - alpha) * start + alpha * goal
            sample = np.clip(
                center + rng.normal(0.0, 0.18, start.shape) * (high - low),
                low,
                high,
            )
        else:
            sample = rng.uniform(low, high)
        index_a = extend(nodes_a, parents_a, sample)
        if index_a is not None:
            while True:
                if time.monotonic() >= deadline:
                    break
                index_b = extend(nodes_b, parents_b, nodes_a[index_a])
                if index_b is None:
                    break
                if np.linalg.norm(nodes_b[index_b] - nodes_a[index_a]) < 1e-9:
                    from_a = chain(nodes_a, parents_a, index_a)
                    from_b = chain(nodes_b, parents_b, index_b)
                    path = from_a + list(reversed(from_b[:-1]))
                    if swapped:
                        path.reverse()
                    break
            if path is not None:
                break
        nodes_a, nodes_b = nodes_b, nodes_a
        parents_a, parents_b = parents_b, parents_a
        swapped = not swapped
    if path is None:
        return None

    shortened = [path[0]]
    index = 0
    while index < len(path) - 1:
        if time.monotonic() >= deadline:
            return None
        candidate = len(path) - 1
        while candidate > index + 1 and not edge_is_valid(path[index], path[candidate]):
            candidate -= 1
        shortened.append(path[candidate])
        index = candidate
    return tuple(tuple(node.tolist()) for node in shortened)


def default_urdf_path(repo_root: Path) -> Path:
    return repo_root / ".deps" / "xr_teleoperate" / "assets" / "g1" / "g1_body29_hand14.urdf"


class G1RightArmIK:
    """Locked-waist G1 right-arm IK adapted from Unitree's pinned CasADi approach.

    The upstream repository is fetched by scripts/dev/fetch-arm-assets.sh so its
    license, meshes, and exact revision remain intact under .deps.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.15,
        discontinuity_limit_rad: float = 0.25,
        joint_limit_margin_rad: float = 0.05,
        translation_weight: float = 50.0,
        orientation_weight: float = 1.0,
    ) -> None:
        try:
            import pinocchio as pin
            from scipy.optimize import least_squares
        except Exception as exc:  # pragma: no cover - host dependency
            raise IKUnavailable(
                "G1 IK requires the arm dependency group (pinocchio and scipy)."
            ) from exc

        self.pin = pin
        self.least_squares = least_squares
        self.casadi = None
        self.cpin = None
        try:
            import casadi
            from pinocchio import casadi as cpin

            self.casadi = casadi
            self.cpin = cpin
        except (ImportError, ModuleNotFoundError):
            # PyPI/uv Pinocchio wheels do not currently ship the optional
            # CasADi bindings. The bounded SciPy backend below uses the same
            # Pinocchio model, objective terms, and acceptance gates.
            pass
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise IKUnavailable(
                f"Pinned G1 URDF missing: {self.urdf_path}. Run scripts/dev/fetch-arm-assets.sh."
            )
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.discontinuity_limit_rad = discontinuity_limit_rad
        if not np.isfinite(translation_weight) or translation_weight <= 0.0:
            raise IKUnavailable("translation_weight must be finite and positive")
        self.translation_weight = float(translation_weight)
        if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
            raise IKUnavailable("orientation_weight must be finite and non-negative")
        self.orientation_weight = float(orientation_weight)

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        unlocked = set(RIGHT_ARM_JOINTS)
        lock_ids = [
            joint_id
            for joint_id in range(1, full_model.njoints)
            if full_model.names[joint_id] not in unlocked
        ]
        reference = pin.neutral(full_model)
        self.collision_model = None
        self.collision_data = None
        try:
            full_collision = pin.buildGeomFromUrdf(
                full_model,
                str(self.urdf_path),
                pin.GeometryType.COLLISION,
                package_dirs=[str(self.urdf_path.parent)],
            )
            reduced = pin.buildReducedModel(
                full_model,
                [full_collision],
                lock_ids,
                reference,
            )
            self.model, geometry_models = reduced
            self.collision_model = geometry_models[0]
            self.collision_model.addAllCollisionPairs()
            self.collision_data = pin.GeometryData(self.collision_model)
        except Exception:
            self.model = pin.buildReducedModel(full_model, lock_ids, reference)
        reduced_joint_order = tuple(self.model.names[1:])
        if self.model.nq != 7 or reduced_joint_order != RIGHT_ARM_JOINTS:
            raise IKUnavailable(
                "Reduced URDF joint order does not match the seven-element right-arm command "
                f"contract: expected {RIGHT_ARM_JOINTS}, got {reduced_joint_order}"
            )
        self.data = self.model.createData()
        self._right_hand_hip_pair_indices = (
            ()
            if self.collision_model is None
            else tuple(
                index
                for index in range(len(self.collision_model.collisionPairs))
                if self._is_right_hand_hip_pair(index)
            )
        )
        self._right_arm_collision_samples: tuple[tuple[int, np.ndarray], ...] = ()
        if self.collision_model is not None:
            samples: list[tuple[int, np.ndarray]] = []
            arm_prefixes = (
                "right_shoulder_",
                "right_elbow_",
                "right_wrist_",
                "right_hand_",
            )
            for geometry_index, geometry_object in enumerate(
                self.collision_model.geometryObjects
            ):
                if not geometry_object.name.startswith(arm_prefixes):
                    continue
                shape = geometry_object.geometry
                try:
                    shape.computeLocalAABB()
                    minimum = np.asarray(shape.aabb_local.min_, dtype=float)
                    maximum = np.asarray(shape.aabb_local.max_, dtype=float)
                except Exception:
                    samples = []
                    break
                if minimum.shape != (3,) or maximum.shape != (3,):
                    samples = []
                    break
                midpoint = 0.5 * (minimum + maximum)
                local_points = [midpoint]
                local_points.extend(
                    np.asarray((x, y, z), dtype=float)
                    for x in (minimum[0], maximum[0])
                    for y in (minimum[1], maximum[1])
                    for z in (minimum[2], maximum[2])
                )
                samples.extend((geometry_index, point) for point in local_points)
            self._right_arm_collision_samples = tuple(samples)
        parent = self.model.getJointId("right_wrist_yaw_joint")
        parent_frame = self.model.getFrameId("right_wrist_yaw_link")
        self.ee_frame = self.model.addFrame(
            pin.Frame(
                "right_pregrasp_frame",
                parent,
                parent_frame,
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        # Frame data must be rebuilt after extending the reduced model.
        self.data = self.model.createData()

        lower = self.model.lowerPositionLimit.copy() + joint_limit_margin_rad
        upper = self.model.upperPositionLimit.copy() - joint_limit_margin_rad
        if np.any(lower >= upper):
            raise IKUnavailable("URDF limits are too narrow for the configured safety margin.")
        self.lower = lower
        self.upper = upper
        self.global_backend = "scipy_least_squares"
        self.local_backend = "pinocchio_analytic_dls"
        if self.casadi is not None and self.cpin is not None:
            self._build_optimizer()
            self.global_backend = "pinocchio_casadi_ipopt"

    def _build_optimizer(self) -> None:
        casadi = self.casadi
        cpin = self.cpin
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        cq = casadi.SX.sym("q", self.model.nq, 1)
        target = casadi.SX.sym("target", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, cq)
        pose = self.cdata.oMf[self.ee_frame]
        translation_error = pose.translation - target[:3, 3]
        rotation_error = cpin.log3(pose.rotation @ target[:3, :3].T)
        self.translation_error = casadi.Function(
            "g1_r_translation_error", [cq, target], [translation_error]
        )
        self.rotation_error = casadi.Function("g1_r_rotation_error", [cq, target], [rotation_error])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_last_q = self.opti.parameter(self.model.nq)
        self.param_target = self.opti.parameter(4, 4)
        cost = (
            self.translation_weight
            * casadi.sumsqr(self.translation_error(self.var_q, self.param_target))
            + self.orientation_weight
            * casadi.sumsqr(self.rotation_error(self.var_q, self.param_target))
            + 0.1 * casadi.sumsqr(self.var_q - self.param_last_q)
            + 0.02 * casadi.sumsqr(self.var_q)
        )
        self.opti.subject_to(self.opti.bounded(self.lower, self.var_q, self.upper))
        self.opti.minimize(cost)
        self.opti.solver(
            "ipopt",
            {
                "expand": True,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": 40,
                "ipopt.tol": 1e-4,
            },
        )

    def forward_kinematics(self, q_rad: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("q_rad must contain seven finite joint positions")
        self.pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self.ee_frame]
        transform = np.eye(4)
        transform[:3, :3] = pose.rotation
        transform[:3, 3] = pose.translation
        return transform

    def shoulder_position(self, q_rad: Sequence[float]) -> np.ndarray:
        """Return the actual right shoulder-pitch origin in torso coordinates.

        Pointing must use this URDF frame, rather than a hand-written shoulder
        estimate.  That keeps the ray consistent with the same model used for
        IK and avoids a systematic over-aim on the tabletop.
        """

        q = np.asarray(q_rad, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("q_rad must contain seven finite joint positions")
        self.pin.framesForwardKinematics(self.model, self.data, q)
        frame_id = self.model.getFrameId("right_shoulder_pitch_joint")
        return np.asarray(self.data.oMf[frame_id].translation, dtype=float).copy()

    def _solve_numerical(self, target: np.ndarray, last_q: np.ndarray) -> np.ndarray:
        sqrt_translation = np.sqrt(self.translation_weight)
        sqrt_orientation = np.sqrt(self.orientation_weight)
        sqrt_smooth = np.sqrt(0.1)
        sqrt_regularization = np.sqrt(0.02)

        def residual(q: np.ndarray) -> np.ndarray:
            self.pin.framesForwardKinematics(self.model, self.data, q)
            pose = self.data.oMf[self.ee_frame]
            translation = sqrt_translation * (pose.translation - target[:3, 3])
            rotation = sqrt_orientation * self.pin.log3(pose.rotation @ target[:3, :3].T)
            return np.concatenate(
                (
                    translation,
                    rotation,
                    sqrt_smooth * (q - last_q),
                    sqrt_regularization * q,
                )
            )

        result = self.least_squares(
            residual,
            np.clip(last_q, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=80,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(result.message)
        return np.asarray(result.x, dtype=float).reshape(7)

    def _collision_pairs(self, q: np.ndarray) -> set[int]:
        if self.collision_model is None or self.collision_data is None:
            return set()
        self.pin.computeCollisions(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
            False,
        )
        return {
            index
            for index, result in enumerate(self.collision_data.collisionResults)
            if result.isCollision() and not self._is_adjacent_g1_pair(index)
        }

    def _is_adjacent_g1_pair(self, pair_index: int) -> bool:
        assert self.collision_model is not None
        pair = self.collision_model.collisionPairs[pair_index]
        first = self.collision_model.geometryObjects[pair.first].name
        second = self.collision_model.geometryObjects[pair.second].name
        # The branded chest/logo mesh has no collision volume on the physical
        # robot.  It sits inside the torso collision mesh and must not reject
        # an otherwise valid shoulder pose.
        if first == "logo_link_0" or second == "logo_link_0":
            return True
        # This is a right-arm-only reduced model.  The left hand is locked at
        # URDF neutral rather than measured, so hand-to-hand contacts are not
        # meaningful constraints for a right-arm pointing trajectory.  Keep
        # every torso/hip/right-arm pair active.
        if (first.startswith("left_hand_") and second.startswith("right_hand_")) or (
            first.startswith("right_hand_") and second.startswith("left_hand_")
        ):
            return True
        return frozenset((first, second)) in _ADJACENT_G1_COLLISION_PAIRS

    def _collision_pair_label(self, pair_index: int) -> str:
        assert self.collision_model is not None
        pair = self.collision_model.collisionPairs[pair_index]
        first = self.collision_model.geometryObjects[pair.first].name
        second = self.collision_model.geometryObjects[pair.second].name
        return f"{first}<->{second}"

    def _collision_pair_names(self, pair_index: int) -> tuple[str, str]:
        assert self.collision_model is not None
        pair = self.collision_model.collisionPairs[pair_index]
        return (
            self.collision_model.geometryObjects[pair.first].name,
            self.collision_model.geometryObjects[pair.second].name,
        )

    def _is_right_hand_hip_pair(self, pair_index: int) -> bool:
        first, second = self._collision_pair_names(pair_index)
        return bool(
            (
                first.startswith("right_hip_")
                and (second.startswith("right_hand_") or second.startswith("right_wrist_"))
            )
            or (
                second.startswith("right_hip_")
                and (first.startswith("right_hand_") or first.startswith("right_wrist_"))
            )
        )

    def _right_hand_hip_distances(
        self,
        q: np.ndarray,
        pair_indices: Sequence[int] | None = None,
    ) -> dict[int, float]:
        assert self.collision_model is not None and self.collision_data is not None
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
        )
        indices = (
            self._right_hand_hip_pair_indices
            if pair_indices is None
            else tuple(int(index) for index in pair_indices)
        )
        return {
            index: float(
                self.pin.computeDistance(
                    self.collision_model,
                    self.collision_data,
                    index,
                ).min_distance
            )
            for index in indices
        }

    def _right_hand_hip_clearance(
        self,
        q: np.ndarray,
        pair_indices: Sequence[int] | None = None,
    ) -> float:
        return min(
            self._right_hand_hip_distances(q, pair_indices).values(),
            default=float("inf"),
        )

    def _colliding_hand_hip_pairs(
        self,
        q: np.ndarray,
        pair_indices: Sequence[int] | None = None,
    ) -> set[int]:
        """Check selected semantic pairs without recomputing every FCL pair."""

        assert self.collision_model is not None and self.collision_data is not None
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
        )
        indices = self._right_hand_hip_pair_indices if pair_indices is None else pair_indices
        return {
            int(index)
            for index in indices
            if self.pin.computeCollision(
                self.collision_model,
                self.collision_data,
                int(index),
            )
        }

    def _hand_hip_edge_is_strictly_clear(
        self,
        begin: np.ndarray,
        end: np.ndarray,
        *,
        step_rad: float = 0.0005,
    ) -> bool:
        steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / step_rad)))
        return all(
            not self._colliding_hand_hip_pairs((1.0 - alpha) * begin + alpha * end)
            for alpha in np.linspace(0.0, 1.0, steps + 1)
        )

    def _hand_hip_edge_is_exit_only(
        self,
        begin: np.ndarray,
        end: np.ndarray,
        initial_pairs: set[int],
        baseline_distances: dict[int, float],
        *,
        step_rad: float = 0.0005,
        numerical_tolerance_m: float = 0.00005,
        require_cleared_end: bool = True,
    ) -> str | None:
        """Permit only shallow measured contacts until they clear, with no re-entry.

        FCL penetration depth is discontinuous while triangle meshes overlap,
        so comparing every sample with the first contact depth rejects valid
        exits.  Bound the entire semantic hand/hip contact family by the same
        absolute 5 mm hard limit used for the measured start pose instead.
        This permits contact to transfer between adjacent hand mesh components;
        collision-free state still latches strict no-re-entry.
        """

        steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / step_rad)))
        escaped = False
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            q = (1.0 - alpha) * begin + alpha * end
            colliding = self._colliding_hand_hip_pairs(q)
            if not colliding:
                escaped = True
                continue
            if escaped:
                return "hand_hip_collision_reentry"
            distances = self._right_hand_hip_distances(q, tuple(sorted(colliding)))
            if any(
                distance < -_MAX_START_ESCAPE_PENETRATION_M - numerical_tolerance_m
                for distance in distances.values()
            ):
                return "hand_hip_penetration_limit"
        if require_cleared_end and (
            not escaped or self._right_hand_hip_clearance(end) < 0.01
        ):
            return "start_collision_not_cleared"
        return None

    def _arm_chain_points(self, q: np.ndarray) -> tuple[np.ndarray, ...]:
        self.pin.framesForwardKinematics(self.model, self.data, q)
        points = [
            np.asarray(self.data.oMi[self.model.getJointId(name)].translation, dtype=float).copy()
            for name in RIGHT_ARM_JOINTS
        ]
        points.append(np.asarray(self.data.oMf[self.ee_frame].translation, dtype=float).copy())
        return tuple(points)

    def _arm_has_support_clearance(
        self,
        q: np.ndarray,
        support_plane: Any | None,
        *,
        minimum_clearance_m: float = 0.05,
    ) -> bool:
        if support_plane is None:
            return True
        if (
            self.collision_model is None
            or self.collision_data is None
            or not self._right_arm_collision_samples
        ):
            return False
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
        )
        # AABBs conservatively enclose every shoulder/elbow/wrist/hand mesh.
        # Checking their transformed corners models the full link envelope,
        # rather than pretending the arm is a zero-radius centerline.
        for geometry_index, local_point in self._right_arm_collision_samples:
            placement = self.collision_data.oMg[geometry_index]
            point = placement.rotation @ local_point + placement.translation
            if hasattr(support_plane, "has_clearance"):
                clear = support_plane.has_clearance(
                    point,
                    minimum_clearance_m=minimum_clearance_m,
                )
            else:
                clear = bool(float(support_plane.signed_distance(point)) >= minimum_clearance_m)
            if not clear:
                return False
        return True

    def _guided_start_escape(
        self,
        start_q: np.ndarray,
        *,
        support_plane: Any | None,
        timeout_s: float = 1.0,
    ) -> tuple[tuple[tuple[float, ...], ...] | None, str | None]:
        """Exit a small measured hand/hip mesh penetration before RRT.

        The initial real collision is not added to an allowed-collision set.
        Only right-hand/right-hip contacts may remain touching, none may exceed
        the 5 mm escape ceiling, and the first collision-free sample latches
        strict mode. The endpoint must have 10 mm modeled clearance.
        """

        initial = self._collision_pairs(start_q)
        if not initial:
            return ((tuple(float(value) for value in start_q),), None)
        if any(not self._is_right_hand_hip_pair(index) for index in initial):
            labels = ",".join(self._collision_pair_label(index) for index in sorted(initial))
            return None, f"start_self_collision:{labels}"
        baseline_distances = self._right_hand_hip_distances(
            start_q,
            tuple(sorted(initial)),
        )
        baseline_clearance = min(baseline_distances.values())
        # More than 5 mm of modeled penetration is not a small mesh-tolerance
        # recovery and must remain an operator-visible hard failure.
        if baseline_clearance < -_MAX_START_ESCAPE_PENETRATION_M:
            return None, f"start_hand_hip_penetration:{baseline_clearance:.5f}"
        deadline = time.monotonic() + timeout_s
        start_tuple = tuple(float(value) for value in start_q)

        def edge_is_exit_only(candidate: np.ndarray) -> bool:
            if self._hand_hip_edge_is_exit_only(
                start_q,
                candidate,
                initial,
                baseline_distances,
            ) is not None:
                return False
            if time.monotonic() >= deadline:
                return False
            samples = max(
                2,
                int(np.ceil(np.max(np.abs(candidate - start_q)) / 0.0025)),
            )
            for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
                q = (1.0 - alpha) * start_q + alpha * candidate
                collisions = self._collision_pairs(q)
                if any(not self._is_right_hand_hip_pair(index) for index in collisions):
                    return False
                if not self._arm_has_support_clearance(q, support_plane):
                    return False
            return not self._collision_pairs(candidate)

        for delta in _G1_HIP_ESCAPE_DELTAS:
            candidate = np.clip(
                start_q + np.asarray(delta, dtype=float),
                self.lower,
                self.upper,
            )
            if edge_is_exit_only(candidate):
                # The bridge accepts at most 0.05 rad between targets.  Keep
                # a margin below that hard gate and retain the already
                # validated straight edge as bounded streaming waypoints.
                steps = max(
                    1,
                    int(
                        np.ceil(
                            np.max(np.abs(candidate - start_q))
                            / _MAX_STREAM_JOINT_STEP_RAD
                        )
                    ),
                )
                return tuple(
                    start_tuple
                    if index == 0
                    else tuple(
                        float(value)
                        for value in (
                            (1.0 - index / steps) * start_q
                            + (index / steps) * candidate
                        )
                    )
                    for index in range(steps + 1)
                ), None

        if time.monotonic() >= deadline:
            return None, "start_collision_escape_timeout"
        return None, "start_collision_escape_unavailable"

    def collision_labels(self, q_rad: Sequence[float]) -> tuple[str, ...]:
        """Return modeled, non-excluded self-collisions for an arm pose."""

        q = np.asarray(q_rad, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return ("invalid_joint_pose",)
        return tuple(
            self._collision_pair_label(index)
            for index in sorted(self._collision_pairs(q))
        )

    def plan_start_collision_escape(
        self,
        start_q_rad: Sequence[float],
        *,
        support_plane: Any | None,
        timeout_s: float = 12.0,
    ) -> IKPathResult:
        """Plan and independently validate a bounded hand-from-hip escape.

        This public entry point is intentionally narrower than general path
        planning: only a shallow right-hand/right-hip contact may be present
        initially, the path may never enter another collision, every link must
        retain support-region clearance, and the endpoint must be at least
        10 mm clear of the hip. Returned waypoints are no more than 0.04 rad
        apart so the bridge can stream them one at a time.
        """

        start_q = np.asarray(start_q_rad, dtype=float)
        if (
            start_q.shape != (7,)
            or not np.all(np.isfinite(start_q))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            return IKPathResult(
                False,
                None,
                float("inf"),
                0.0,
                "invalid_start_collision_escape",
            )
        path, error = self._guided_start_escape(
            start_q,
            support_plane=support_plane,
            timeout_s=timeout_s,
        )
        if path is None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                0.0,
                error or "start_collision_escape_unavailable",
            )
        if len(path) < 2:
            return IKPathResult(
                False,
                None,
                0.0,
                0.0,
                "start_pose_collision_free",
            )
        validation_error = self.validate_joint_path(
            path,
            support_plane=support_plane,
            edge_step_rad=0.0025,
            deadline_s=time.monotonic() + timeout_s,
        )
        if validation_error is not None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                0.0,
                validation_error,
            )
        return IKPathResult(True, path, 0.0, 0.0, None)

    def _candidate_q(self, target: np.ndarray, last_q: np.ndarray) -> np.ndarray:
        if self.casadi is not None and self.cpin is not None:
            self.opti.set_initial(self.var_q, np.clip(last_q, self.lower, self.upper))
            self.opti.set_value(self.param_last_q, last_q)
            self.opti.set_value(self.param_target, target)
            solution = self.opti.solve()
            return np.asarray(solution.value(self.var_q), dtype=float).reshape(7)
        return self._solve_numerical(target, last_q)

    def _translation_jacobian(
        self,
        q: np.ndarray,
        *,
        backend: str,
        current_position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the world-aligned end-effector translation Jacobian.

        The analytic path applies the useful part of YuehChuan/unitreeG1_ik's
        MuJoCo DLS approach to our authoritative Pinocchio model. Keeping the
        model and safety layer unchanged avoids the reference repository's
        incorrect mirrored shoulder limit and missing collision/limit gates.
        """

        if backend == "analytic":
            jacobian6 = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            jacobian = np.asarray(jacobian6[:3, :], dtype=float)
        elif backend == "finite_difference":
            position = (
                self.forward_kinematics(q)[:3, 3]
                if current_position is None
                else np.asarray(current_position, dtype=float)
            )
            epsilon = 1e-4
            jacobian = np.empty((3, 7), dtype=float)
            for index in range(7):
                perturbed = q.copy()
                direction = 1.0
                if perturbed[index] + epsilon > self.upper[index]:
                    direction = -1.0
                perturbed[index] += direction * epsilon
                candidate_position = self.forward_kinematics(perturbed)[:3, 3]
                jacobian[:, index] = (
                    candidate_position - position
                ) / (direction * epsilon)
        else:
            raise ValueError(
                "jacobian_backend must be 'analytic' or 'finite_difference'"
            )
        if jacobian.shape != (3, 7) or not np.all(np.isfinite(jacobian)):
            raise RuntimeError("translation Jacobian is invalid")
        return jacobian

    def plan_guided_clearance(
        self,
        start_q_rad: Sequence[float],
        *,
        support_plane: Any | None,
        lift_m: float = 0.25,
        forward_m: float = 0.06,
        position_tolerance_m: float = 0.006,
        max_steps_per_phase: int = 100,
    ) -> IKPathResult:
        """Escape the hip, lift, then move forward using validated local IK."""

        start_q = np.asarray(start_q_rad, dtype=float)
        if (
            start_q.shape != (7,)
            or not np.all(np.isfinite(start_q))
            or not np.isfinite(lift_m)
            or not np.isfinite(forward_m)
            or lift_m <= 0.0
            or forward_m < 0.0
            or not 0.001 <= position_tolerance_m <= 0.02
            or max_steps_per_phase <= 0
        ):
            return IKPathResult(False, None, float("inf"), 0.0, "invalid_guided_clearance")

        escape_path, escape_error = self._guided_start_escape(
            start_q,
            support_plane=support_plane,
            # Pinocchio collision queries are substantially slower on the
            # GB10 ARM build.  Keep the same finite candidate set and safety
            # predicates, but allow enough wall time to evaluate them.
            timeout_s=12.0,
        )
        if escape_path is None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                0.0,
                escape_error or "start_collision_escape_unavailable",
            )

        path = list(escape_path)
        q = np.asarray(path[-1], dtype=float)
        for phase_name, delta_xyz in (
            # Keep the long XR finger meshes behind the tabletop's 7 cm
            # uncertainty margin during the subsequent vertical lift.
            ("retract", np.asarray((-0.015, 0.0, 0.0), dtype=float)),
            ("lift", np.asarray((0.0, 0.0, lift_m), dtype=float)),
            ("forward", np.asarray((forward_m, 0.0, 0.0), dtype=float)),
        ):
            if float(np.linalg.norm(delta_xyz)) <= 1e-9:
                continue
            target = self.forward_kinematics(q)
            target[:3, 3] += delta_xyz
            previous_error = float("inf")
            for _ in range(max_steps_per_phase):
                error = float(
                    np.linalg.norm(
                        target[:3, 3] - self.forward_kinematics(q)[:3, 3]
                    )
                )
                if error <= position_tolerance_m:
                    break
                result = self.solve_local_translation(
                    target,
                    q,
                    support_plane=support_plane,
                    maximum_joint_step_rad=0.025,
                    # Clearance is a large Cartesian relocation.  Prefer the
                    # shoulder/elbow chain so the long finger meshes do not
                    # swing forward through the tabletop boundary while the
                    # hand is still below it.
                    inverse_joint_cost_weights=(
                        (2.0, 2.0, 1.5, 2.0, 0.01, 0.01, 0.01)
                        if phase_name == "lift"
                        else (1.0, 1.0, 1.0, 1.2, 0.6, 3.0, 3.0)
                    ),
                )
                if not result.ok or result.q_rad is None:
                    return IKPathResult(
                        False,
                        None,
                        error,
                        0.0,
                        f"guided_{phase_name}:{result.reason or 'ik_failed'}",
                    )
                candidate = np.asarray(result.q_rad, dtype=float)
                candidate_error = float(
                    np.linalg.norm(
                        target[:3, 3] - self.forward_kinematics(candidate)[:3, 3]
                    )
                )
                if candidate_error >= min(error, previous_error) - 1e-5:
                    return IKPathResult(
                        False,
                        None,
                        candidate_error,
                        0.0,
                        f"guided_{phase_name}:no_progress",
                    )
                q = candidate
                path.append(tuple(float(value) for value in q))
                previous_error = candidate_error
            else:
                return IKPathResult(
                    False,
                    None,
                    previous_error,
                    0.0,
                    f"guided_{phase_name}:step_limit",
                )

        return IKPathResult(True, tuple(path), 0.0, 0.0, None)

    def _ik_seed_variants(self, start_q: np.ndarray) -> tuple[np.ndarray, ...]:
        """Offer the optimizer bounded G1 arm postures, not one local branch.

        A single smooth seed can repeatedly fold the elbow toward the torso.
        These alternatives bias shoulder yaw, elbow, and wrist posture while
        retaining the measured arm pose as the actual route start.  They are
        only seeds for IK; collision checking still vets the solved goal and
        every joint-space edge.
        """

        offsets = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.40, -0.50, 0.0, -0.25, 0.20),
            (0.0, 0.0, -0.40, -0.50, 0.0, 0.25, -0.20),
            (0.0, 0.15, 0.35, -0.35, 0.20, -0.35, 0.25),
            (0.0, -0.15, -0.35, -0.35, -0.20, 0.35, -0.25),
            (0.15, 0.10, 0.20, -0.60, 0.25, -0.40, 0.30),
            (-0.15, -0.10, -0.20, -0.60, -0.25, 0.40, -0.30),
        )
        variants: list[np.ndarray] = []
        for offset in offsets:
            candidate = np.clip(start_q + np.asarray(offset, dtype=float), self.lower, self.upper)
            if not any(np.allclose(candidate, existing, atol=1e-6) for existing in variants):
                variants.append(candidate)
        return tuple(variants)

    def _pose_errors(self, target: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        self.pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self.ee_frame]
        position_error = float(np.linalg.norm(pose.translation - target[:3, 3]))
        orientation_error = float(np.linalg.norm(self.pin.log3(pose.rotation @ target[:3, :3].T)))
        return position_error, orientation_error

    def solve_with_collision_detour(
        self,
        target_transform: np.ndarray,
        start_q_rad: Sequence[float],
        *,
        enforce_orientation: bool = True,
        support_plane: Any | None = None,
        planning_timeout_s: float = 20.0,
        max_iterations: int = 7000,
    ) -> IKPathResult:
        """Solve a goal pose, then route around transient self-collisions.

        The start pose's existing mesh overlaps are the only collision pairs
        tolerated. Every edge returned by the planner is swept at 0.04 rad or
        finer before it can be sent through the separate 0.05-rad ROS guard.
        """

        target = np.asarray(target_transform, dtype=float)
        start_q = np.asarray(start_q_rad, dtype=float)
        if target.shape != (4, 4) or start_q.shape != (7,):
            return IKPathResult(False, None, float("inf"), float("inf"), "invalid_shape")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(start_q)):
            return IKPathResult(False, None, float("inf"), float("inf"), "non_finite")
        if not np.isfinite(planning_timeout_s) or planning_timeout_s <= 0.0 or max_iterations <= 0:
            return IKPathResult(
                False,
                None,
                float("inf"),
                float("inf"),
                "invalid_planning_budget",
            )
        if self.collision_model is None or self.collision_data is None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                float("inf"),
                "collision_model_unavailable",
            )
        started_at = time.monotonic()
        deadline = started_at + planning_timeout_s
        escape_path, escape_error = self._guided_start_escape(
            start_q,
            support_plane=support_plane,
            timeout_s=min(3.0, planning_timeout_s * 0.40),
        )
        if escape_path is None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                float("inf"),
                escape_error or "start_collision_escape_unavailable",
            )
        route_start_q = np.asarray(escape_path[-1], dtype=float)
        if self._collision_pairs(route_start_q):
            return IKPathResult(
                False,
                None,
                float("inf"),
                float("inf"),
                "start_collision_not_cleared",
            )
        start_xyz = self.forward_kinematics(start_q)[:3, 3]
        last_failure = (float("inf"), float("inf"), "solver_failed")
        goals: list[tuple[np.ndarray, float, float]] = []
        for seed_q in self._ik_seed_variants(start_q):
            if time.monotonic() >= deadline:
                break
            try:
                goal_q = self._candidate_q(target, seed_q)
            except Exception:
                continue
            if float(np.max(np.abs(goal_q - start_q))) > 1.20:
                last_failure = (float("inf"), float("inf"), "goal_joint_delta_too_large")
                continue
            position_error, orientation_error = self._pose_errors(target, goal_q)
            if position_error > self.position_tolerance_m:
                last_failure = (position_error, orientation_error, "position_error")
                continue
            if (
                enforce_orientation
                and self.orientation_weight > 0.0
                and orientation_error > self.orientation_tolerance_rad
            ):
                last_failure = (position_error, orientation_error, "orientation_error")
                continue
            goal_collisions = self._collision_pairs(goal_q)
            if goal_collisions:
                labels = ",".join(
                    self._collision_pair_label(index) for index in sorted(goal_collisions)
                )
                last_failure = (position_error, orientation_error, f"goal_self_collision:{labels}")
                continue
            if not self._arm_has_support_clearance(goal_q, support_plane):
                last_failure = (
                    position_error,
                    orientation_error,
                    "goal_support_region_clearance",
                )
                continue
            if not any(np.allclose(goal_q, item[0], atol=1e-5) for item in goals):
                goals.append((goal_q, position_error, orientation_error))

        goals.sort(key=lambda item: float(np.linalg.norm(item[0] - route_start_q)))
        # A close joint-space goal is not necessarily the easiest collision
        # branch to reach.  In particular, the two closest G1 solutions can
        # both fold the elbow toward the hip while a slightly farther
        # wrist/elbow branch has a simple route over the table.  Keep every
        # independently validated IK branch available to the bounded planner.
        selected_goals = goals
        for goal_index, (goal_q, position_error, orientation_error) in enumerate(selected_goals):
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                last_failure = (
                    position_error,
                    orientation_error,
                    "collision_detour_timeout",
                )
                break

            goal_xyz = self.forward_kinematics(goal_q)[:3, 3]
            corridor_margin = np.array([0.50, 0.40, 0.50, 0.70, 0.50, 0.50, 0.60])
            lower = np.maximum(
                self.lower,
                np.minimum(route_start_q, goal_q) - corridor_margin,
            )
            upper = np.minimum(
                self.upper,
                np.maximum(route_start_q, goal_q) + corridor_margin,
            )

            def state_is_valid(q: np.ndarray) -> bool:
                if self._collision_pairs(q):
                    return False
                xyz = self.forward_kinematics(q)[:3, 3]
                if xyz[0] < min(start_xyz[0], goal_xyz[0]) - 0.04:
                    return False
                if xyz[2] < min(start_xyz[2], goal_xyz[2]) - 0.03:
                    return False
                if not -0.48 <= xyz[1] <= 0.15:
                    return False
                return self._arm_has_support_clearance(q, support_plane)

            validation_reserve = min(2.5, max(0.5, remaining * 0.25))
            # Give each branch enough time to leave a collision boundary, but
            # cap a bad branch so later elbow/wrist alternatives still run.
            candidate_budget = min(3.0, remaining - validation_reserve)
            if candidate_budget <= 0.05:
                last_failure = (
                    position_error,
                    orientation_error,
                    "collision_detour_timeout",
                )
                break
            path = collision_aware_joint_path(
                route_start_q,
                goal_q,
                lower,
                upper,
                state_is_valid,
                edge_step_rad=0.005,
                extension_step_rad=0.14,
                max_iterations=max_iterations,
                planning_timeout_s=candidate_budget,
                seed=31 + goal_index,
                guided_sampling=True,
            )
            if path is None:
                last_failure = (position_error, orientation_error, "collision_detour_unavailable")
                continue
            combined = (*escape_path[:-1], *path)
            # Search may legitimately consume its whole bounded budget just as
            # it finds a route. Never convert that into an automatic
            # validation_timeout by handing the dense final sweep an expired
            # deadline; validation receives its own small, disarmed-only cap.
            validation_deadline = time.monotonic() + 3.0
            validation_error = self.validate_joint_path(
                combined,
                support_plane=support_plane,
                edge_step_rad=0.0025,
                deadline_s=validation_deadline,
            )
            if validation_error:
                last_failure = (
                    position_error,
                    orientation_error,
                    f"planned_path_invalid:{validation_error}",
                )
                continue
            return IKPathResult(True, combined, position_error, orientation_error)
        position_error, orientation_error, reason = last_failure
        return IKPathResult(False, None, position_error, orientation_error, reason)

    def solve(
        self,
        target_transform: np.ndarray,
        last_q_rad: Sequence[float],
        *,
        support_plane: Any | None = None,
    ) -> IKResult:
        target = np.asarray(target_transform, dtype=float)
        last_q = np.asarray(last_q_rad, dtype=float)
        if target.shape != (4, 4) or last_q.shape != (7,):
            return IKResult(False, None, float("inf"), float("inf"), "invalid_shape")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(last_q)):
            return IKResult(False, None, float("inf"), float("inf"), "non_finite")
        try:
            q = self._candidate_q(target, last_q)
        except Exception:
            return IKResult(False, None, float("inf"), float("inf"), "solver_failed")

        maximum_delta = float(np.max(np.abs(q - last_q)))
        if maximum_delta > self.discontinuity_limit_rad:
            return IKResult(
                False,
                None,
                float("inf"),
                float("inf"),
                "discontinuous:"
                f"max_delta={maximum_delta:.4f}:"
                f"limit={self.discontinuity_limit_rad:.4f}",
            )
        if self.collision_model is None or self.collision_data is None:
            return IKResult(False, None, float("inf"), float("inf"), "collision_model_unavailable")
        # Explicit adjacent mesh exclusions are handled by _collision_pairs.
        # Never turn a measured hand/hip collision into a realtime-session ACM.
        start_collisions = self._collision_pairs(last_q)
        if start_collisions:
            labels = ",".join(
                self._collision_pair_label(index) for index in sorted(start_collisions)
            )
            return IKResult(
                False,
                None,
                float("inf"),
                float("inf"),
                f"start_self_collision:{labels}",
            )
        sweep_steps = max(2, int(np.ceil(maximum_delta / 0.0025)))
        for alpha in np.linspace(0.0, 1.0, sweep_steps + 1):
            swept_q = (1.0 - alpha) * last_q + alpha * q
            collisions = self._collision_pairs(swept_q)
            if collisions:
                labels = ",".join(self._collision_pair_label(index) for index in sorted(collisions))
                return IKResult(
                    False,
                    None,
                    float("inf"),
                    float("inf"),
                    f"self_collision:{labels}:path_fraction={alpha:.3f}",
                )
            if not self._arm_has_support_clearance(swept_q, support_plane):
                return IKResult(
                    False,
                    None,
                    float("inf"),
                    float("inf"),
                    "link_support_region_clearance",
                )
        position_error, orientation_error = self._pose_errors(target, q)
        if position_error > self.position_tolerance_m:
            return IKResult(False, None, position_error, orientation_error, "position_error")
        if self.orientation_weight > 0.0 and orientation_error > self.orientation_tolerance_rad:
            return IKResult(False, None, position_error, orientation_error, "orientation_error")
        return IKResult(True, tuple(float(value) for value in q), position_error, orientation_error)

    def solve_local_translation(
        self,
        target_transform: np.ndarray,
        last_q_rad: Sequence[float],
        *,
        support_plane: Any | None = None,
        maximum_joint_step_rad: float = 0.040,
        damping: float = 0.01,
        inverse_joint_cost_weights: Sequence[float] = (
            1.0,
            1.0,
            1.0,
            1.2,
            0.6,
            3.0,
            3.0,
        ),
        validate_path: bool = True,
        jacobian_backend: str = "analytic",
    ) -> IKResult:
        """Compute one fast warm-started Cartesian servo step.

        An analytic Pinocchio Jacobian implements the same damped-least-squares
        core as the evaluated MuJoCo reference without changing the canonical
        robot model. Wrist pitch/yaw receive useful redundancy weight, while
        the returned edge still passes the complete collision, table-clearance,
        corridor, and joint-limit validation. The previous finite-difference
        implementation remains selectable for offline A/B regression tests.
        """

        target = np.asarray(target_transform, dtype=float)
        last_q = np.asarray(last_q_rad, dtype=float)
        if (
            target.shape != (4, 4)
            or last_q.shape != (7,)
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(last_q))
            or not math.isfinite(maximum_joint_step_rad)
            or not 0.0 < maximum_joint_step_rad <= 0.05
            or not math.isfinite(damping)
            or damping <= 0.0
            or jacobian_backend not in ("analytic", "finite_difference")
            or len(inverse_joint_cost_weights) != 7
            or not all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in inverse_joint_cost_weights
            )
        ):
            return IKResult(False, None, float("inf"), float("inf"), "invalid_local_step")
        current = self.forward_kinematics(last_q)
        error = target[:3, 3] - current[:3, 3]
        initial_error = float(np.linalg.norm(error))
        if initial_error <= 1e-6:
            if validate_path:
                validation_error = self.validate_joint_path(
                    (last_q, last_q),
                    support_plane=support_plane,
                    edge_step_rad=0.010,
                )
                if validation_error is not None:
                    return IKResult(False, None, 0.0, 0.0, validation_error)
            return IKResult(True, tuple(float(value) for value in last_q), 0.0, 0.0)

        try:
            jacobian = self._translation_jacobian(
                last_q,
                backend=jacobian_backend,
                current_position=current[:3, 3],
            )
        except (RuntimeError, ValueError):
            return IKResult(False, None, initial_error, 0.0, "local_jacobian_invalid")

        # Weighted minimum-norm solution. The short hand offset makes wrist
        # pitch/yaw genuinely useful for translation; prefer them when safe
        # instead of forcing every correction through shoulder and elbow.
        inverse_joint_cost = np.diag(tuple(float(value) for value in inverse_joint_cost_weights))
        system = jacobian @ inverse_joint_cost @ jacobian.T
        system += np.eye(3) * (damping * damping)
        try:
            delta_q = inverse_joint_cost @ jacobian.T @ np.linalg.solve(system, error)
        except np.linalg.LinAlgError:
            return IKResult(False, None, initial_error, 0.0, "local_jacobian_singular")
        maximum_delta = float(np.max(np.abs(delta_q)))
        if maximum_delta > maximum_joint_step_rad:
            delta_q *= maximum_joint_step_rad / maximum_delta

        last_reason = "local_step_invalid"
        for scale in (1.0, 0.5, 0.25):
            candidate = np.clip(last_q + delta_q * scale, self.lower, self.upper)
            candidate_error = float(
                np.linalg.norm(self.forward_kinematics(candidate)[:3, 3] - target[:3, 3])
            )
            if candidate_error >= initial_error - 1e-5:
                last_reason = "local_step_no_progress"
                continue
            if not validate_path:
                return IKResult(
                    True,
                    tuple(float(value) for value in candidate),
                    candidate_error,
                    0.0,
                )
            validation_error = self.validate_joint_path(
                (last_q, candidate),
                support_plane=support_plane,
                # Four swept intervals keep every maximum 0.040-rad realtime
                # edge collision/table checked without returning to the much
                # slower full-route sampling cadence.
                edge_step_rad=0.010,
            )
            if validation_error is None:
                return IKResult(
                    True,
                    tuple(float(value) for value in candidate),
                    candidate_error,
                    0.0,
                )
            last_reason = validation_error
        return IKResult(False, None, initial_error, 0.0, last_reason)

    def validate_joint_path(
        self,
        knots: Sequence[Sequence[float]],
        *,
        support_plane: Any | None = None,
        edge_step_rad: float = 0.0025,
        deadline_s: float | None = None,
    ) -> str | None:
        """Validate a short, already-planned path without invoking RRT or IK.

        A small measured hand/hip penetration may only occur in an exit-only
        prefix that stays within the hard penetration ceiling and becomes
        collision-free.
        Once clear, every real collision pair is forbidden.
        """

        if (
            len(knots) < 2
            or not math.isfinite(edge_step_rad)
            or edge_step_rad <= 0.0
            or (deadline_s is not None and not math.isfinite(deadline_s))
        ):
            return "invalid_joint_path"
        path = tuple(np.asarray(knot, dtype=float) for knot in knots)
        if any(knot.shape != (7,) or not np.all(np.isfinite(knot)) for knot in path):
            return "invalid_joint_path"
        if self.collision_model is None or self.collision_data is None:
            return "collision_model_unavailable"
        if any(np.any(knot < self.lower) or np.any(knot > self.upper) for knot in path):
            return "joint_limit"

        initial_collisions = self._collision_pairs(path[0])
        if any(not self._is_right_hand_hip_pair(index) for index in initial_collisions):
            labels = ",".join(
                self._collision_pair_label(index) for index in sorted(initial_collisions)
            )
            return f"start_self_collision:{labels}"
        baseline_distances = (
            self._right_hand_hip_distances(path[0], tuple(sorted(initial_collisions)))
            if initial_collisions
            else {}
        )
        initial_clearance = min(baseline_distances.values(), default=float("inf"))
        if initial_clearance < -_MAX_START_ESCAPE_PENETRATION_M:
            return f"start_hand_hip_penetration:{initial_clearance:.5f}"
        escaping = bool(initial_collisions)
        started_escaping = escaping
        for begin, end in zip(path, path[1:]):
            if deadline_s is not None and time.monotonic() >= deadline_s:
                return "validation_timeout"
            maximum_delta = float(np.max(np.abs(end - begin)))
            if escaping:
                escape_error = self._hand_hip_edge_is_exit_only(
                    begin,
                    end,
                    initial_collisions,
                    baseline_distances,
                    require_cleared_end=False,
                )
                if escape_error:
                    return escape_error
            elif maximum_delta > 0.025 and not self._hand_hip_edge_is_strictly_clear(
                begin,
                end,
            ):
                return "hand_hip_collision_reentry"
            steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / edge_step_rad)))
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                if deadline_s is not None and time.monotonic() >= deadline_s:
                    return "validation_timeout"
                q = (1.0 - alpha) * begin + alpha * end
                collisions = self._collision_pairs(q)
                if escaping:
                    forbidden = {
                        index for index in collisions if not self._is_right_hand_hip_pair(index)
                    }
                    if forbidden:
                        labels = ",".join(
                            self._collision_pair_label(index) for index in sorted(forbidden)
                        )
                        return f"self_collision:{labels}:path_fraction={alpha:.3f}"
                elif collisions:
                    labels = ",".join(
                        self._collision_pair_label(index) for index in sorted(collisions)
                    )
                    return f"self_collision:{labels}:path_fraction={alpha:.3f}"
                hand_xyz = self.forward_kinematics(q)[:3, 3]
                if not -0.48 <= hand_xyz[1] <= 0.15:
                    return "safe_corridor"
                if not self._arm_has_support_clearance(q, support_plane):
                    return "link_support_region_clearance"
            if escaping:
                escaping = bool(self._collision_pairs(end))
        if escaping or (
            started_escaping and self._right_hand_hip_clearance(path[-1]) < 0.01
        ):
            return "start_collision_not_cleared"
        return None
