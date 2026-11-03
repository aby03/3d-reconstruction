"""PLY files: OpenMVS's dense point cloud (per-vertex view lists), plain binary point clouds, mesh statistics."""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# PLY scalar type -> numpy little-endian dtype
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}
POINT_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
     ("red", "u1"), ("green", "u1"), ("blue", "u1")]
)


def _read_header(data: bytes) -> tuple[int, int, list[tuple[str, str]], list[tuple[str, str, str]]]:
    """(vertex count, data offset, scalar properties, list properties) of a binary little-endian vertex-only PLY."""
    end = data.find(b"end_header\n")
    if not data.startswith(b"ply\n") or end < 0:
        raise ValueError("not a PLY file")
    lines = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary little-endian PLY is supported")
    count, scalars, lists, element = 0, [], [], None
    for line in lines:
        words = line.split()
        if words[:1] == ["element"]:
            element = words[1]
            if element == "vertex":
                count = int(words[2])
            elif int(words[2]):
                raise ValueError(f"unexpected element {element!r}")
        elif words[:1] == ["property"] and element == "vertex":
            if words[1] == "list":
                lists.append((words[4], _PLY_TYPES[words[2]], _PLY_TYPES[words[3]]))
            elif lists:
                raise ValueError("scalar vertex properties after a list property are not supported")
            else:
                scalars.append((words[2], _PLY_TYPES[words[1]]))
    return count, end + len("end_header\n"), scalars, lists


def _record_starts(buf: bytes, start: int, count: int, fixed: int, lists: list[tuple[str, str, str]]) -> np.ndarray:
    """Byte offset of every vertex record. The list lengths make the records variable-sized, so this is the one
    sequential scan (Python ints from `bytes` indexing: a uint8 count can't overflow the offset maths)."""
    if any(np.dtype(c).itemsize != 1 for _, c, _ in lists):
        raise ValueError("only uint8 list counts are supported")
    sizes = [np.dtype(item).itemsize for _, _, item in lists]
    starts, pos = [], start
    for _ in range(count):
        starts.append(pos)
        pos += fixed
        for size in sizes:
            pos += 1 + size * buf[pos]
    if pos > len(buf):
        raise ValueError("PLY is shorter than its vertex records")
    return np.asarray(starts, dtype=np.int64)


def _gather(raw: np.ndarray, starts: np.ndarray, offset: int, dtype: str) -> np.ndarray:
    size = np.dtype(dtype).itemsize
    return raw[starts[:, None] + offset + np.arange(size)].copy().view(dtype).ravel()


def read_openmvs_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """xyz (N,3) float64, normals (N,3) float64, rgb (N,3) uint8 and views per point (N,) int of OpenMVS's
    `scene_dense.ply` (x y z, red green blue, nx ny nz, then the view_indices / view_weights lists)."""
    buf = Path(path).read_bytes()
    count, start, scalars, lists = _read_header(buf)
    offsets, pos = {}, 0
    for name, dtype in scalars:
        offsets[name] = (pos, dtype)
        pos += np.dtype(dtype).itemsize
    if missing := {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"} - set(offsets):
        raise ValueError(f"{path}: missing vertex properties {sorted(missing)}")
    if count == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, int)
    raw = np.frombuffer(buf, dtype=np.uint8)
    starts = _record_starts(buf, start, count, pos, lists) if lists else start + pos * np.arange(count)

    def columns(*names: str) -> np.ndarray:
        return np.stack([_gather(raw, starts, *offsets[n]) for n in names], axis=1)

    views = raw[starts + pos].astype(int) if lists else np.zeros(count, int)
    xyz, normals = columns("x", "y", "z").astype(float), columns("nx", "ny", "nz").astype(float)
    return xyz, normals, columns("red", "green", "blue").astype(np.uint8), views


def write_points(path: Path, xyz: np.ndarray, normals: np.ndarray, rgb: np.ndarray) -> Path:
    """Binary little-endian PLY with x y z nx ny nz (float32) and red green blue (uint8) per vertex."""
    data = np.empty(len(xyz), dtype=POINT_DTYPE)
    for i, name in enumerate("xyz"):
        data[name] = xyz[:, i]
        data[f"n{name}"] = normals[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        data[name] = rgb[:, i]
    properties = "".join(
        f"property {'uchar' if POINT_DTYPE[n].kind == 'u' else 'float'} {n}\n" for n in POINT_DTYPE.names
    )
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n{properties}end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return path


def read_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load(path, process=False, force="mesh")


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Vertex and face counts, number of face-connected components and the share of faces in the largest one."""
    num_faces = len(mesh.faces)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=num_faces)
    sizes = np.bincount(labels) if num_faces else np.zeros(0, int)
    return {
        "vertices": len(mesh.vertices),
        "faces": num_faces,
        "components": len(sizes),
        "largest_component_frac": float(sizes.max() / num_faces) if num_faces else 0.0,
    }


def select_openmvs_points(src: Path, dst: Path, keep: np.ndarray) -> Path:
    """Write the vertices of the OpenMVS point cloud `src` where `keep` (N,) is true to `dst`, records (and their
    view lists) copied unchanged, so OpenMVS can read it back as a dense point cloud."""
    buf = Path(src).read_bytes()
    count, start, scalars, lists = _read_header(buf)
    if len(keep) != count:
        raise ValueError(f"{src}: {count} vertices, mask of {len(keep)}")
    fixed = sum(np.dtype(dtype).itemsize for _, dtype in scalars)
    starts = _record_starts(buf, start, count, fixed, lists) if lists else start + fixed * np.arange(count)
    ends = np.append(starts[1:], len(buf))
    header = buf[: start - len(b"end_header\n")].decode("ascii")
    header = header.replace(f"element vertex {count}", f"element vertex {int(keep.sum())}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write((header + "end_header\n").encode("ascii"))
        for i in np.flatnonzero(keep):
            f.write(buf[starts[i] : ends[i]])
    return dst
