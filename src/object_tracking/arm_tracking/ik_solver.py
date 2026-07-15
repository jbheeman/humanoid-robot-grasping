from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .joints import RIGHT_ARM_JOINT_NAMES


XR_TELEOPERATE_REVISION = "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
UNITREE_ROS_REVISION = "d96d8f63ae17a7108d4f7229c00ef875ba7129c9"
RIGHT_ARM_JOINTS = RIGHT_ARM_JOINT_NAMES

# These collision meshes overlap at the G1's factory shoulder articulation
# range.  They are kinematic neighbours, not an arm-through-torso route.  The
# remaining mesh pairs (hand/hip, forearm/torso, table clearance, joint limits)
# stay active in every planning query.
_ADJACENT_G1_COLLISION_PAIRS = {
    frozenset(("torso_link_0", "right_shoulder_pitch_link_0")),
    frozenset(("torso_link_0", "right_shoulder_roll_link_0")),
    frozenset(("torso_link_0", "right_shoulder_yaw_link_0")),
    frozenset(("torso_link_0", "right_elbow_link_0")),
}


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
    max_iterations: int = 2500,
    seed: int = 7,
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
    ):
        raise ValueError("invalid collision-aware joint path inputs")

    def edge_is_valid(begin: np.ndarray, end: np.ndarray) -> bool:
        steps = max(1, int(np.ceil(np.max(np.abs(end - begin)) / edge_step_rad)))
        return all(
            state_is_valid((1.0 - alpha) * begin + alpha * end)
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]
        )

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

    def extend(
        nodes: list[np.ndarray], parents: list[int], target: np.ndarray
    ) -> int | None:
        parent = nearest(nodes, target)
        begin = nodes[parent]
        delta = target - begin
        distance = float(np.linalg.norm(delta))
        end = target if distance <= extension_step_rad else begin + delta / distance * extension_step_rad
        if not edge_is_valid(begin, end):
            return None
        nodes.append(end)
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
        sample = goal if rng.random() < 0.15 else rng.uniform(low, high)
        index_a = extend(nodes_a, parents_a, sample)
        if index_a is not None:
            while True:
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
        if self.casadi is not None and self.cpin is not None:
            self._build_optimizer()

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
            rotation = sqrt_orientation * self.pin.log3(
                pose.rotation @ target[:3, :3].T
            )
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
        if (
            first.startswith("left_hand_") and second.startswith("right_hand_")
        ) or (
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

    def _candidate_q(self, target: np.ndarray, last_q: np.ndarray) -> np.ndarray:
        if self.casadi is not None and self.cpin is not None:
            self.opti.set_initial(self.var_q, np.clip(last_q, self.lower, self.upper))
            self.opti.set_value(self.param_last_q, last_q)
            self.opti.set_value(self.param_target, target)
            solution = self.opti.solve()
            return np.asarray(solution.value(self.var_q), dtype=float).reshape(7)
        return self._solve_numerical(target, last_q)

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
        orientation_error = float(
            np.linalg.norm(self.pin.log3(pose.rotation @ target[:3, :3].T))
        )
        return position_error, orientation_error

    def solve_with_collision_detour(
        self,
        target_transform: np.ndarray,
        start_q_rad: Sequence[float],
        *,
        enforce_orientation: bool = True,
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
        if self.collision_model is None or self.collision_data is None:
            return IKPathResult(
                False,
                None,
                float("inf"),
                float("inf"),
                "collision_model_unavailable",
            )
        allowed_collisions = self._collision_pairs(start_q)
        start_xyz = self.forward_kinematics(start_q)[:3, 3]
        last_failure = (float("inf"), float("inf"), "solver_failed")
        for seed_q in self._ik_seed_variants(start_q):
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
            introduced_at_goal = self._collision_pairs(goal_q) - allowed_collisions
            if introduced_at_goal:
                labels = ",".join(
                    self._collision_pair_label(index) for index in sorted(introduced_at_goal)
                )
                last_failure = (position_error, orientation_error, f"goal_self_collision:{labels}")
                continue

            goal_xyz = self.forward_kinematics(goal_q)[:3, 3]
            corridor_margin = np.array([0.50, 0.40, 0.50, 0.70, 0.50, 0.50, 0.60])
            lower = np.maximum(self.lower, np.minimum(start_q, goal_q) - corridor_margin)
            upper = np.minimum(self.upper, np.maximum(start_q, goal_q) + corridor_margin)

            def state_is_valid(q: np.ndarray) -> bool:
                if self._collision_pairs(q) - allowed_collisions:
                    return False
                xyz = self.forward_kinematics(q)[:3, 3]
                if xyz[0] < min(start_xyz[0], goal_xyz[0]) - 0.04:
                    return False
                if xyz[2] < min(start_xyz[2], goal_xyz[2]) - 0.03:
                    return False
                return bool(-0.48 <= xyz[1] <= 0.15)

            path = collision_aware_joint_path(start_q, goal_q, lower, upper, state_is_valid)
            if path is None:
                last_failure = (position_error, orientation_error, "collision_detour_unavailable")
                continue
            # The measured-pose seed is always first.  Once a collision-free
            # route is found, execute it rather than spending control time
            # searching for cosmetically different redundant postures.
            return IKPathResult(True, path, position_error, orientation_error)
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
        # Mesh URDFs commonly contain persistent adjacent-link overlaps. Treat
        # those present at the measured start pose as the baseline and reject
        # only collision pairs introduced by the candidate sweep.
        baseline_collisions = self._collision_pairs(last_q)
        for alpha in np.linspace(0.0, 1.0, 12):
            swept_q = (1.0 - alpha) * last_q + alpha * q
            introduced = self._collision_pairs(swept_q) - baseline_collisions
            if introduced:
                labels = ",".join(
                    self._collision_pair_label(index)
                    for index in sorted(introduced)
                )
                return IKResult(
                    False,
                    None,
                    float("inf"),
                    float("inf"),
                    f"self_collision:{labels}:path_fraction={alpha:.3f}",
                )
        if support_plane is not None:
            self.pin.framesForwardKinematics(self.model, self.data, q)
            for joint_name in RIGHT_ARM_JOINTS:
                joint_id = self.model.getJointId(joint_name)
                point = self.data.oMi[joint_id].translation
                if float(support_plane.signed_distance(point)) < 0.05:
                    return IKResult(False, None, float("inf"), float("inf"), "link_plane_clearance")
        position_error, orientation_error = self._pose_errors(target, q)
        if position_error > self.position_tolerance_m:
            return IKResult(False, None, position_error, orientation_error, "position_error")
        if (
            self.orientation_weight > 0.0
            and orientation_error > self.orientation_tolerance_rad
        ):
            return IKResult(False, None, position_error, orientation_error, "orientation_error")
        return IKResult(True, tuple(float(value) for value in q), position_error, orientation_error)
