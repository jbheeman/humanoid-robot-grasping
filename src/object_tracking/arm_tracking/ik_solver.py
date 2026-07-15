from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .joints import RIGHT_ARM_JOINT_NAMES


XR_TELEOPERATE_REVISION = "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
UNITREE_ROS_REVISION = "d96d8f63ae17a7108d4f7229c00ef875ba7129c9"
RIGHT_ARM_JOINTS = RIGHT_ARM_JOINT_NAMES


class IKUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class IKResult:
    ok: bool
    q_rad: tuple[float, ...] | None
    position_error_m: float
    orientation_error_rad: float
    reason: str | None = None


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
            50.0 * casadi.sumsqr(self.translation_error(self.var_q, self.param_target))
            + casadi.sumsqr(self.rotation_error(self.var_q, self.param_target))
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

    def _solve_numerical(self, target: np.ndarray, last_q: np.ndarray) -> np.ndarray:
        sqrt_translation = np.sqrt(50.0)
        sqrt_smooth = np.sqrt(0.1)
        sqrt_regularization = np.sqrt(0.02)

        def residual(q: np.ndarray) -> np.ndarray:
            self.pin.framesForwardKinematics(self.model, self.data, q)
            pose = self.data.oMf[self.ee_frame]
            translation = sqrt_translation * (pose.translation - target[:3, 3])
            rotation = self.pin.log3(pose.rotation @ target[:3, :3].T)
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
            if result.isCollision()
        }

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
            if self.casadi is not None and self.cpin is not None:
                self.opti.set_initial(self.var_q, np.clip(last_q, self.lower, self.upper))
                self.opti.set_value(self.param_last_q, last_q)
                self.opti.set_value(self.param_target, target)
                solution = self.opti.solve()
                q = np.asarray(solution.value(self.var_q), dtype=float).reshape(7)
            else:
                q = self._solve_numerical(target, last_q)
        except Exception:
            return IKResult(False, None, float("inf"), float("inf"), "solver_failed")

        if np.max(np.abs(q - last_q)) > self.discontinuity_limit_rad:
            return IKResult(False, None, float("inf"), float("inf"), "discontinuous")
        if self.collision_model is None or self.collision_data is None:
            return IKResult(False, None, float("inf"), float("inf"), "collision_model_unavailable")
        # Mesh URDFs commonly contain persistent adjacent-link overlaps. Treat
        # those present at the measured start pose as the baseline and reject
        # only collision pairs introduced by the candidate sweep.
        baseline_collisions = self._collision_pairs(last_q)
        for alpha in np.linspace(0.0, 1.0, 12):
            swept_q = (1.0 - alpha) * last_q + alpha * q
            if self._collision_pairs(swept_q) - baseline_collisions:
                return IKResult(False, None, float("inf"), float("inf"), "self_collision")
        self.pin.framesForwardKinematics(self.model, self.data, q)
        if support_plane is not None:
            for joint_name in RIGHT_ARM_JOINTS:
                joint_id = self.model.getJointId(joint_name)
                point = self.data.oMi[joint_id].translation
                if float(support_plane.signed_distance(point)) < 0.05:
                    return IKResult(False, None, float("inf"), float("inf"), "link_plane_clearance")
        pose = self.data.oMf[self.ee_frame]
        position_error = float(np.linalg.norm(pose.translation - target[:3, 3]))
        orientation_error = float(np.linalg.norm(self.pin.log3(pose.rotation @ target[:3, :3].T)))
        if position_error > self.position_tolerance_m:
            return IKResult(False, None, position_error, orientation_error, "position_error")
        if orientation_error > self.orientation_tolerance_rad:
            return IKResult(False, None, position_error, orientation_error, "orientation_error")
        return IKResult(True, tuple(float(value) for value in q), position_error, orientation_error)
