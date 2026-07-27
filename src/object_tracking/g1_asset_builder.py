"""Deterministic, dependency-light G1 URDF/STL to browser GLB builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request
import xml.etree.ElementTree as ET


THREE_VERSION = "0.170.0"
TOOL_VERSION = "g1-visual-assets-v1"
VENDOR_URLS = {
    "three.module.js": f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.js",
    "GLTFLoader.js": f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/loaders/GLTFLoader.js",
    "OrbitControls.js": f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js",
    "BufferGeometryUtils.js": f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/utils/BufferGeometryUtils.js",
}


def _numbers(value: str | None, count: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    result = tuple(float(item) for item in value.split())
    if len(result) != count or not all(math.isfinite(item) for item in result):
        raise ValueError(f"expected {count} finite numbers, got {value!r}")
    return result


def _quaternion(rpy: tuple[float, float, float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _safe_mesh_path(urdf: Path, filename: str) -> Path:
    raw = filename.removeprefix("package://")
    candidates = [urdf.parent / raw]
    if raw.startswith("assets/g1/"):
        candidates.append(urdf.parent.parent.parent / raw)
    resolved_root = urdf.parent.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise ValueError(f"mesh path escapes the URDF asset root or is missing: {filename}")


def _stl_triangles(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    data = path.read_bytes()
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            for index in range(count):
                values = struct.unpack_from("<12fH", data, 84 + index * 50)
                triangles.append((values[3:6], values[6:9], values[9:12]))
            return triangles
    words = data.decode("utf-8", errors="strict").split()
    vertices = []
    for index, word in enumerate(words):
        if word == "vertex" and index + 3 < len(words):
            vertices.append(tuple(float(item) for item in words[index + 1 : index + 4]))
    if len(vertices) % 3:
        raise ValueError(f"invalid STL vertex count: {path}")
    return [tuple(vertices[index : index + 3]) for index in range(0, len(vertices), 3)]


def _simplify(triangles, ratio: float = 0.15):
    if not triangles:
        raise ValueError("mesh contains no triangles")
    wanted = max(1, int(math.ceil(len(triangles) * ratio)))
    if wanted >= len(triangles):
        return triangles
    # Even deterministic sampling preserves the full source ordering/range and
    # avoids adding a heavyweight geometry stack to the isolated robot setup.
    return [triangles[(index * len(triangles)) // wanted] for index in range(wanted)]


class _Glb:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []

    def _append(self, payload: bytes, target: int) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(payload)
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target}
        )
        return len(self.buffer_views) - 1

    def mesh(self, name: str, triangles) -> int:
        positions = [coordinate for triangle in triangles for vertex in triangle for coordinate in vertex]
        vertices = [positions[index : index + 3] for index in range(0, len(positions), 3)]
        pos_view = self._append(struct.pack(f"<{len(positions)}f", *positions), 34962)
        pos_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": [min(vertex[axis] for vertex in vertices) for axis in range(3)],
                "max": [max(vertex[axis] for vertex in vertices) for axis in range(3)],
            }
        )
        indices = list(range(len(vertices)))
        index_view = self._append(struct.pack(f"<{len(indices)}I", *indices), 34963)
        index_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {"attributes": {"POSITION": pos_accessor}, "indices": index_accessor, "material": 0}
                ],
            }
        )
        return len(self.meshes) - 1

    def encode(self, nodes: list[dict], roots: list[int]) -> bytes:
        document = {
            "asset": {"version": "2.0", "generator": TOOL_VERSION},
            "scene": 0,
            "scenes": [{"nodes": roots}],
            "nodes": nodes,
            "meshes": self.meshes,
            "materials": [
                {
                    "name": "neutral",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.58, 0.61, 0.64, 0.16],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.9,
                    },
                    "alphaMode": "BLEND",
                    "doubleSided": True,
                }
            ],
            "buffers": [{"byteLength": len(self.binary)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
        }
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        binary = bytes(self.binary) + b"\0" * (-len(self.binary) % 4)
        length = 12 + 8 + len(encoded) + 8 + len(binary)
        return (
            struct.pack("<4sII", b"glTF", 2, length)
            + struct.pack("<I4s", len(encoded), b"JSON")
            + encoded
            + struct.pack("<I4s", len(binary), b"BIN\0")
            + binary
        )


def _source_hash(urdf: Path, meshes: list[Path]) -> str:
    digest = hashlib.sha256(TOOL_VERSION.encode())
    tool_sources = [
        Path(__file__),
        Path(__file__).resolve().parents[2] / "scripts/shared/g1_visualizer.js",
    ]
    for path in tool_sources:
        digest.update(path.read_bytes())
    for path in [urdf, *sorted(set(meshes))]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _vendor(output: Path, vendor_dir: Path | None) -> None:
    for name, url in VENDOR_URLS.items():
        destination = output / name
        source = None if vendor_dir is None else vendor_dir / name
        if source is not None and source.is_file():
            shutil.copyfile(source, destination)
        elif destination.is_file():
            pass
        else:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    destination.write_bytes(response.read())
            except OSError as exc:
                raise RuntimeError(
                    f"could not obtain pinned {name}; retry with network access or --vendor-dir"
                ) from exc
        # Three's examples use a package import that browsers cannot resolve
        # without a bundler/import-map. Keep the generated directory standalone.
        if name != "three.module.js":
            destination.write_text(
                destination.read_text(encoding="utf-8")
                .replace("from 'three'", "from './three.module.js'")
                .replace('from "three"', 'from "./three.module.js"')
                .replace("from '../utils/BufferGeometryUtils.js'", "from './BufferGeometryUtils.js'"),
                encoding="utf-8",
            )
    (output / "LICENSES.txt").write_text(
        "G1 model and meshes: xr_teleoperate, see source LICENSE (BSD-3-Clause).\n"
        f"Three.js {THREE_VERSION}: Copyright three.js authors, MIT License.\n",
        encoding="utf-8",
    )
    visualizer = Path(__file__).resolve().parents[2] / "scripts/shared/g1_visualizer.js"
    shutil.copyfile(visualizer, output / "g1_visualizer.js")


def build_assets(urdf_path: Path, output: Path, *, vendor_dir: Path | None = None) -> dict:
    urdf = urdf_path.resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    root = ET.parse(urdf).getroot()
    visuals = []
    mesh_paths = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.attrib.get("filename"):
                continue
            path = _safe_mesh_path(urdf, mesh.attrib["filename"])
            mesh_paths.append(path)
            visuals.append((link.attrib["name"], visual, mesh, path))
    digest = _source_hash(urdf, mesh_paths)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        required = [output / name for name in ("g1.glb", *VENDOR_URLS, "LICENSES.txt", "g1_visualizer.js")]
        if existing.get("source_hash") == digest and all(path.is_file() for path in required):
            return {**existing, "cache_hit": True}
    output.mkdir(parents=True, exist_ok=True)
    glb = _Glb()
    links = [link.attrib["name"] for link in root.findall("link")]
    nodes: list[dict] = [{"name": name, "extras": {"kind": "link"}, "children": []} for name in links]
    link_nodes = {name: index for index, name in enumerate(links)}
    original_triangles = simplified_triangles = 0
    for link_name, visual, mesh, path in visuals:
        triangles = _stl_triangles(path)
        simplified = _simplify(triangles)
        original_triangles += len(triangles)
        simplified_triangles += len(simplified)
        mesh_index = glb.mesh(f"{link_name}:{path.name}", simplified)
        origin = visual.find("origin")
        node = {
            "name": f"{link_name}__visual_{len(nodes)}",
            "mesh": mesh_index,
            "translation": list(_numbers(None if origin is None else origin.attrib.get("xyz"), 3, (0, 0, 0))),
            "rotation": _quaternion(_numbers(None if origin is None else origin.attrib.get("rpy"), 3, (0, 0, 0))),
            "scale": list(_numbers(mesh.attrib.get("scale"), 3, (1, 1, 1))),
            "extras": {"kind": "visual", "link": link_name},
        }
        nodes.append(node)
        nodes[link_nodes[link_name]]["children"].append(len(nodes) - 1)
    child_links = set()
    joint_manifest = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name, child_name = parent.attrib["link"], child.attrib["link"]
        if parent_name not in link_nodes or child_name not in link_nodes:
            continue
        origin = joint.find("origin")
        xyz = _numbers(None if origin is None else origin.attrib.get("xyz"), 3, (0, 0, 0))
        rpy = _numbers(None if origin is None else origin.attrib.get("rpy"), 3, (0, 0, 0))
        axis_element = joint.find("axis")
        axis = _numbers(None if axis_element is None else axis_element.attrib.get("xyz"), 3, (1, 0, 0))
        joint_node = {
            "name": joint.attrib["name"],
            "translation": list(xyz),
            "rotation": _quaternion(rpy),
            "children": [link_nodes[child_name]],
            "extras": {"kind": "joint", "axis": list(axis), "type": joint.attrib.get("type", "fixed")},
        }
        nodes.append(joint_node)
        nodes[link_nodes[parent_name]]["children"].append(len(nodes) - 1)
        child_links.add(child_name)
        joint_manifest.append(
            {"name": joint.attrib["name"], "parent_link": parent_name, "child_link": child_name, "axis": list(axis), "type": joint.attrib.get("type", "fixed")}
        )
    roots = [link_nodes[name] for name in links if name not in child_links]
    (output / "g1.glb").write_bytes(glb.encode(nodes, roots))
    _vendor(output, vendor_dir)
    source_license = urdf.parent.parent.parent / "LICENSE"
    if source_license.is_file():
        shutil.copyfile(source_license, output / "XR_TELEOPERATE_LICENSE.txt")
    manifest = {
        "schema_version": 1,
        "model_id": "unitree-g1-body29-hand14-xr-teleoperate",
        "source_revision": "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6",
        "source_hash": digest,
        "tool_version": TOOL_VERSION,
        "three_version": THREE_VERSION,
        "model": "g1.glb",
        "joint_hierarchy": joint_manifest,
        "link_names": links,
        "original_triangles": original_triangles,
        "simplified_triangles": simplified_triangles,
        "simplification_ratio": 0 if original_triangles == 0 else round(simplified_triangles / original_triangles, 6),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if size > 12 * 1024 * 1024:
        raise RuntimeError(f"generated visual bundle is {size / 1024 / 1024:.1f} MB; limit is 12 MB")
    return {**manifest, "cache_hit": False, "bundle_bytes": size}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build ignored, offline G1 browser assets")
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path, help="single output override (tests/custom builds)")
    parser.add_argument("--vendor-dir", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    urdf = args.urdf or root / ".deps/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    output = args.output or root / "scripts/gb10/web/visual"
    report = build_assets(urdf, output, vendor_dir=args.vendor_dir)
    state = "cache hit" if report["cache_hit"] else "built"
    print(f"{output}: {state} ({report.get('bundle_bytes', 'unchanged')} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
