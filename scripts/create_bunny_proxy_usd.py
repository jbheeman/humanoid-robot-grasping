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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

project_root = os.environ.get("G1_BUNNY_PROJECT_ROOT")
if not project_root:
    raise SystemExit("Set G1_BUNNY_PROJECT_ROOT to the humanoid-robot-grasping checkout")
sys.path.insert(0, str(Path(project_root) / "src"))

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from g1_bunny_vla.bunny_proxy import DEFAULT_BUNNY_PROXY  # noqa: E402


def _material(stage: Usd.Stage, path: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", UsdShade.Tokens.color3f).Set(Gf.Vec3f(0.82, 0.67, 0.54))
    shader.CreateInput("roughness", UsdShade.Tokens.float).Set(0.88)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _sphere(stage: Usd.Stage, path: str, position: tuple[float, float, float], scale: tuple[float, float, float], material: UsdShade.Material) -> None:
    shape = UsdGeom.Sphere.Define(stage, path)
    shape.CreateRadiusAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdShade.MaterialBindingAPI(shape).Bind(material)


def _ear(stage: Usd.Stage, path: str, position: tuple[float, float, float], material: UsdShade.Material) -> None:
    shape = UsdGeom.Capsule.Define(stage, path)
    shape.CreateRadiusAttr(0.012)
    shape.CreateHeightAttr(0.064)
    shape.CreateAxisAttr(UsdGeom.Tokens.z)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdShade.MaterialBindingAPI(shape).Bind(material)


def main() -> int:
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

    material = _material(stage, "/Bunny/Looks/PlushTan")
    # The torso drives the measured 15 cm x 7 cm envelope.  The head and ears
    # make the profile grasp-relevant while keeping the total height at 14 cm.
    _sphere(stage, "/Bunny/Geometry/Torso", (0.0, 0.0, 0.038), (0.075, 0.035, 0.038), material)
    _sphere(stage, "/Bunny/Geometry/Head", (0.042, 0.0, 0.081), (0.038, 0.031, 0.031), material)
    _ear(stage, "/Bunny/Geometry/LeftEar", (0.049, 0.018, 0.116), material)
    _ear(stage, "/Bunny/Geometry/RightEar", (0.049, -0.018, 0.116), material)

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    print(f"Wrote rigid-frame bunny proxy to {args.output}")
    print(f"mass={spec.mass_kg:.3f} kg, envelope={spec.depth_m:.3f} x {spec.width_m:.3f} x {spec.height_m:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
