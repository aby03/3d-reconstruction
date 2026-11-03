"""Binary glTF (GLB) of the textured mesh: Y up, vertices welded with smooth normals, the JPEG atlases embedded
unchanged, double-sided PBR materials, and a check of the written file.

trimesh writes the scene; its JPEG path re-encodes the atlases (PIL's default quality), so each material gets a
1x1 placeholder image whose colour names its atlas, and the placeholders are swapped for the atlas bytes after the
export.
"""

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from recon.core.frames import R_GW
from recon.export import GlbOptions
from recon.io.tables import write_bytes
from recon.export.obj import read_mtllib
from recon.mesh.texturing import read_mtl_maps

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def load_textured(obj: Path) -> trimesh.Scene:
    """The textured OBJ as a scene with one geometry per material (atlas), vertices as in the file."""
    return trimesh.load(obj, force="scene", process=False)


def scene_bounds(scene: trimesh.Scene) -> np.ndarray:
    """(2, 3) min/max of every geometry's vertices (the scene graph is the identity for loaded OBJ/GLB files)."""
    vertices = np.concatenate([np.asarray(g.vertices) for g in scene.geometry.values()])
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


def rotate_bounds(bounds: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Axis-aligned bounds of the box `bounds` rotated by `R` (a signed permutation keeps them exact)."""
    corners = np.array([[bounds[i, 0], bounds[j, 1], bounds[k, 2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    rotated = corners @ R.T
    return np.stack([rotated.min(axis=0), rotated.max(axis=0)])


def weld(geometries: list[trimesh.Trimesh]) -> None:
    """In place: merge the vertices with the same position and UV (OpenMVS writes three per face) and set smooth
    vertex normals computed on the mesh welded on position only, over all geometries, so neither UV seams nor atlas
    boundaries show as shading seams."""
    for geometry in geometries:
        geometry.merge_vertices(merge_tex=False, merge_norm=True)
    vertices = np.concatenate([np.asarray(g.vertices) for g in geometries])
    offsets = np.cumsum([0] + [len(g.vertices) for g in geometries])
    faces = np.concatenate([np.asarray(g.faces) + o for g, o in zip(geometries, offsets)])
    unique, inverse = trimesh.grouping.unique_rows(vertices)
    normals = trimesh.Trimesh(vertices[unique], inverse[faces], process=False).vertex_normals[inverse]
    for geometry, start, end in zip(geometries, offsets[:-1], offsets[1:]):
        geometry.vertex_normals = normals[start:end]


def _placeholder(index: int) -> Image.Image:
    return Image.new("RGB", (1, 1), (index & 255, (index >> 8) & 255, 0))


def _placeholder_index(data: bytes) -> int:
    r, g, _ = Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0))
    return r + (g << 8)


def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """JSON and BIN chunk of a GLB, after checking the header, the chunk types and that the lengths add up."""
    if len(data) < 12:
        raise ValueError("GLB shorter than its header")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise ValueError(f"not a glTF 2 binary: magic {magic!r}, version {version}")
    if length != len(data):
        raise ValueError(f"GLB header length {length} != file size {len(data)}")
    chunks, pos = [], 12
    while pos < len(data):
        size, kind = struct.unpack_from("<II", data, pos)
        if pos + 8 + size > len(data):
            raise ValueError("GLB chunk runs past the end of the file")
        chunks.append((kind, data[pos + 8 : pos + 8 + size]))
        pos += 8 + size
    if not chunks or chunks[0][0] != CHUNK_JSON or (len(chunks) > 1 and chunks[1][0] != CHUNK_BIN):
        raise ValueError("GLB chunks must be JSON, then BIN")
    tree = json.loads(chunks[0][1])
    return tree, chunks[1][1] if len(chunks) > 1 else b""


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


def write_glb_bytes(tree: dict[str, Any], binary: bytes) -> bytes:
    text = _pad(json.dumps(tree, separators=(",", ":")).encode(), b" ")
    binary = _pad(binary, b"\0")
    chunks = struct.pack("<II", len(text), CHUNK_JSON) + text + struct.pack("<II", len(binary), CHUNK_BIN) + binary
    return struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, 12 + len(chunks)) + chunks


def _swap_images(glb: bytes, sources: list[Path]) -> bytes:
    """Replace every placeholder image with the bytes of the atlas it names; the buffer views are repacked."""
    tree, binary = read_glb(glb)
    views = tree["bufferViews"]
    blobs = [binary[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]] for v in views]
    for image in tree.get("images", []):
        source = sources[_placeholder_index(blobs[image["bufferView"]])]
        blobs[image["bufferView"]] = source.read_bytes()
        image["mimeType"] = MIME[source.suffix.lower()]
    out = bytearray()
    for view, blob in zip(views, blobs):
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"], view["byteLength"] = len(out), len(blob)
        out += blob
    tree["buffers"] = [{"byteLength": len(out)}]
    return write_glb_bytes(tree, bytes(out))


def write_glb(obj: Path, out: Path, options: GlbOptions | None = None) -> dict[str, Any]:
    """`out` (GLB) from the textured OBJ `obj` in world W: vertices rotated by R_GW (no node transform) and welded
    with smooth normals, one PBR material per atlas (base colour texture, metallic 0, roughness 1, opaque,
    double-sided per `options`).
    Returns the source's face count and W bounds."""
    options = options or GlbOptions()
    scene = load_textured(obj)
    bounds_w = scene_bounds(scene)
    maps = read_mtl_maps(obj.parent / read_mtllib(obj))
    sources: list[Path] = []
    for geometry in scene.geometry.values():
        material = geometry.visual.material
        if material.name not in maps:
            raise ValueError(f"{obj.name}: material {material.name!r} has no map_Kd")
        sources.append(obj.parent / maps[material.name])
        geometry.vertices = np.asarray(geometry.vertices) @ R_GW.T
        geometry.visual.material = trimesh.visual.material.PBRMaterial(
            name=material.name,
            baseColorTexture=_placeholder(len(sources) - 1),
            baseColorFactor=[255, 255, 255, 255],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=options.double_sided,
        )
    weld(list(scene.geometry.values()))
    glb = _swap_images(scene.export(file_type="glb", include_normals=True), sources)
    write_bytes(glb, out)
    return {"faces": sum(len(g.faces) for g in scene.geometry.values()), "bounds_w": bounds_w}


def validate_glb(path: Path, faces: int, bounds_w: np.ndarray, diag: float) -> dict[str, Any]:
    """Check the GLB `path` against its source (face count, W bounds; `diag` scales the bounds tolerance): valid
    header and chunks, every image and buffer embedded (no `uri`), normals on every primitive, materials textured,
    and the reloaded scene with `faces` faces and the bounds R_GW · `bounds_w`. Raises ValueError on the first
    failure."""
    data = path.read_bytes()
    tree, binary = read_glb(data)
    if any("uri" in item for key in ("images", "buffers") for item in tree.get(key, [])):
        raise ValueError(f"{path.name}: external uri in images or buffers")
    if (buffers := tree.get("buffers", [])) and buffers[0]["byteLength"] > len(binary):
        raise ValueError(f"{path.name}: BIN chunk shorter than the buffer")
    for mesh in tree.get("meshes", []):
        if any("NORMAL" not in primitive["attributes"] for primitive in mesh["primitives"]):
            raise ValueError(f"{path.name}: mesh {mesh.get('name')!r} has no vertex normals")
    images = tree.get("images", [])
    if not images or any("bufferView" not in image for image in images):
        raise ValueError(f"{path.name}: images missing or not in a buffer view")
    for material in tree.get("materials", []):
        if "baseColorTexture" not in material.get("pbrMetallicRoughness", {}):
            raise ValueError(f"{path.name}: material {material.get('name')!r} has no base colour texture")
    scene = trimesh.load(path, force="scene")
    num_faces = sum(len(g.faces) for g in scene.geometry.values())
    if num_faces != faces:
        raise ValueError(f"{path.name}: {num_faces} faces, the source has {faces}")
    expected, bounds = rotate_bounds(np.asarray(bounds_w), R_GW), scene_bounds(scene)
    if np.abs(bounds - expected).max() > 1e-4 * diag:
        raise ValueError(f"{path.name}: bounds {bounds.tolist()} != R_GW of the source bounds {expected.tolist()}")
    textures = []
    for image in images:
        view = tree["bufferViews"][image["bufferView"]]
        offset = view.get("byteOffset", 0)
        textures.append(list(Image.open(io.BytesIO(binary[offset : offset + view["byteLength"]])).size))
    return {
        "faces": num_faces,
        "vertices": sum(len(g.vertices) for g in scene.geometry.values()),
        "textures": textures,
        "bytes": len(data),
        "embedded": True,
    }
