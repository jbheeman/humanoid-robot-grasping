#!/usr/bin/env python3
"""Create a rigid-frame bunny plush proxy USD using Isaac Sim's Python.

The model uses a compound collider: torso, head, and two ears. Its measured
envelope is 15 cm front-to-back, 7 cm side-to-side, and 14 cm high; total mass
is 150 g. Run with Isaac Sim's Python, not the conversion environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

project_root = os.environ.get("G1_BUNNY_PROJECT_ROOT")
if not project_root:
    raise SystemExit("Set G1_BUNNY_PROJECT_ROOT to the humanoid-robot-grasping checkout")
sys.path.insert(0, str(Path(project_root) / "src"))

TAN = (0.82, 0.67, 0.54)
CREAM = (0.95, 0.91, 0.83)
PINK = (0.96, 0.53, 0.65)


def _sphere(stage, path, position, scale, color=TAN, collision=True) -> None:
    shape = UsdGeom.Sphere.Define(stage, path)
    shape.CreateRadiusAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*scale))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def _ear(stage, path, position, color=TAN, collision=True) -> None:
    shape = UsdGeom.Capsule.Define(stage, path)
    shape.CreateRadiusAttr(0.010)
    # USD capsule height is the cylindrical span; with the hemispherical caps
    # this keeps the ears below the measured 14 cm total height.
    shape.CreateHeightAttr(0.032)
    shape.CreateAxisAttr(UsdGeom.Tokens.z)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def main() -> int:
    global Gf, Usd, UsdGeom, UsdPhysics
    simulation_app = AppLauncher(args).app
    try:
        # Isaac Sim registers USD bindings only after its application starts.
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
        from g1_bunny_vla.bunny_proxy import DEFAULT_BUNNY_PROXY  # noqa: E402

        spec = DEFAULT_BUNNY_PROXY
        spec.validate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(args.output))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, "/Bunny")
        root.GetPrim().SetMetadata("kind", "component")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass.CreateMassAttr().Set(spec.mass_kg)

        # Upright profile from the supplied reference: rounded haunches, chest,
        # head, front paws, and long ears.  Collision parts stay inside the
        # measured 15 cm (X) x 7 cm (Y) x 14 cm (Z) envelope.
        _sphere(stage, "/Bunny/Geometry/RearBody", (-0.010, 0.0, 0.038), (0.065, 0.035, 0.038))
        _sphere(stage, "/Bunny/Geometry/Chest", (0.020, 0.0, 0.064), (0.040, 0.030, 0.038))
        _sphere(stage, "/Bunny/Geometry/Head", (0.045, 0.0, 0.096), (0.030, 0.029, 0.028))
        _sphere(stage, "/Bunny/Geometry/LeftRearPaw", (-0.035, 0.020, 0.016), (0.034, 0.014, 0.016))
        _sphere(stage, "/Bunny/Geometry/RightRearPaw", (-0.035, -0.020, 0.016), (0.034, 0.014, 0.016))
        _sphere(stage, "/Bunny/Geometry/LeftFrontPaw", (0.035, 0.020, 0.028), (0.017, 0.012, 0.024))
        _sphere(stage, "/Bunny/Geometry/RightFrontPaw", (0.035, -0.020, 0.028), (0.017, 0.012, 0.024))
        _ear(stage, "/Bunny/Geometry/LeftEar", (0.018, 0.017, 0.113))
        _ear(stage, "/Bunny/Geometry/RightEar", (0.018, -0.017, 0.113))
        _sphere(stage, "/Bunny/Visual/Muzzle", (0.073, 0.0, 0.091), (0.008, 0.022, 0.016), CREAM, False)
        _sphere(stage, "/Bunny/Visual/Nose", (0.080, 0.0, 0.096), (0.004, 0.006, 0.004), PINK, False)

        stage.SetDefaultPrim(root.GetPrim())
        stage.GetRootLayer().Save()
        print(f"Wrote rigid-frame bunny proxy to {args.output}", flush=True)
        print(
            f"mass={spec.mass_kg:.3f} kg, envelope={spec.depth_m:.3f} x "
            f"{spec.width_m:.3f} x {spec.height_m:.3f} m",
            flush=True,
        )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
