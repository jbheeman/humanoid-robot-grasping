from pathlib import Path

import pytest

from object_tracking.g1_asset_builder import build_assets


def _fixture(tmp_path: Path, *, escaping: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "assets" / "g1"
    meshes = root / "meshes"
    meshes.mkdir(parents=True)
    triangles = []
    for index in range(20):
        triangles.append(
            f"facet normal 0 0 1 outer loop vertex {index} 0 0 vertex {index} 1 0 vertex {index} 0 1 endloop endfacet"
        )
    (meshes / "tiny.STL").write_text("solid tiny\n" + "\n".join(triangles) + "\nendsolid tiny\n")
    mesh_name = "../../outside.STL" if escaping else "meshes/tiny.STL"
    urdf = root / "tiny.urdf"
    urdf.write_text(
        f"""<robot name="tiny">
        <link name="base"><visual><geometry><mesh filename="{mesh_name}"/></geometry></visual></link>
        <link name="tip"/>
        <joint name="test_joint" type="revolute"><parent link="base"/><child link="tip"/><origin xyz="0 0 1"/><axis xyz="0 1 0"/></joint>
        </robot>"""
    )
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "three.module.js").write_text("export const REVISION='fixture';\n")
    (vendor / "GLTFLoader.js").write_text("import {x} from 'three'; export class GLTFLoader {}\n")
    (vendor / "OrbitControls.js").write_text("import {x} from 'three'; export class OrbitControls {}\n")
    (vendor / "BufferGeometryUtils.js").write_text("import {x} from 'three'; export const toTrianglesDrawMode=()=>{};\n")
    return urdf, vendor


def test_asset_builder_is_deterministic_simplified_attributed_and_cached(tmp_path) -> None:
    urdf, vendor = _fixture(tmp_path)
    output = tmp_path / "output"
    first = build_assets(urdf, output, vendor_dir=vendor)
    manifest_bytes = (output / "manifest.json").read_bytes()
    glb_bytes = (output / "g1.glb").read_bytes()
    second = build_assets(urdf, output, vendor_dir=vendor)
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert (output / "g1.glb").read_bytes() == glb_bytes
    assert first["original_triangles"] == 20
    assert first["simplified_triangles"] == 3
    assert first["joint_hierarchy"][0]["name"] == "test_joint"
    assert "xr_teleoperate" in (output / "LICENSES.txt").read_text()
    assert "./three.module.js" in (output / "GLTFLoader.js").read_text()
    assert glb_bytes[:4] == b"glTF"


def test_asset_builder_rejects_mesh_path_escape(tmp_path) -> None:
    urdf, vendor = _fixture(tmp_path, escaping=True)
    with pytest.raises(ValueError, match="escapes"):
        build_assets(urdf, tmp_path / "output", vendor_dir=vendor)
