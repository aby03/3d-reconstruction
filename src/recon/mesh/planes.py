"""Planar regularization of a room mesh in world W (Z up): floor, ceiling and walls are detected from the face
normals (Manhattan-first: ±Z plus the dominant wall directions), their vertices are snapped onto the fitted planes,
vertices near two or three planes go onto the intersection line or corner, and the rest is Taubin-smoothed."""

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymeshlab
import scipy.sparse as sp
from scipy import ndimage
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from recon.io.ply import read_mesh
from recon.io.tables import write_json
from recon.mesh.cleanup import save

AZIMUTH_BIN_DEG = 1.0
ZONE_FACTOR = 2.0  # a vertex can join a plane when within this many tau of the plane's region
MIN_REGION_SHARE = 0.25  # regions below this share of the minimum plane area are dropped
MIN_PAIR_ANGLE_DEG = 30.0  # planes closer in angle than this don't form an edge
FLIP_PASSES = 50  # moves of the vertices of flipped faces are undone until none is flipped, at most this often
TAUBIN_STEPS, TAUBIN_LAMBDA, TAUBIN_MU = 5, 0.5, -0.53
WALL_RMS_MIN_AREA_FRAC = 0.01  # walls below this share of the area don't count for wall_rms_max_frac
LABELS = ("floor", "wall", "ceiling", "other")


@dataclass(slots=True)
class PlaneOptions:
    enabled: bool = True
    snap_dist: float | None = None  # tau; None = 0.005 * diag (arbitrary units) or 0.02 (metres)
    normal_tol_deg: float = 20.0
    merge_angle_deg: float = 3.0
    manhattan_snap_deg: float = 5.0
    min_plane_area_frac: float = 0.005
    straighten_edges: bool = True
    max_snap_move_factor: float = 1.5
    smooth_nonplanar: bool = True
    fill_patch_weight: float = 0.5  # fill patches (mono depth) count this much in plane detection, so they cannot tilt a plane
    bend_removal: bool = False  # move the vertices of a bent wall onto its plane (see `_remove_bend`)
    wave_cell_frac: float = 0.008  # bend grid cell, as a fraction of the diagonal
    relief_tol_frac: float = 0.0015  # a vertex this far (fraction of the diagonal) from the smoothed bend is relief and stays
    relief_min_frac: float = 0.005  # relief pieces smaller than this (fraction of the diagonal) are noise
    bend_max_tau: float = 6.0  # no vertex moves further than this many tau


@dataclass(slots=True)
class Plane:
    id: int
    label: str
    normal: list[float]  # unit, n . x = d in W
    d: float
    area: float
    area_frac: float
    faces: int
    rms_before: float
    rms_after: float
    axis_angle_deg: float  # between the fitted normal and its detected direction


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _wall_directions(normals: np.ndarray, area: np.ndarray, options: PlaneOptions) -> list[np.ndarray]:
    """Horizontal directions with at least `min_plane_area_frac` of the area within `normal_tol_deg`: peaks of the
    area-weighted azimuth histogram of the near-horizontal face normals, snapped onto ±X/±Y when close."""
    tol = options.normal_tol_deg
    horizontal = np.abs(normals[:, 2]) < np.sin(np.radians(tol))
    azimuth = np.degrees(np.arctan2(normals[horizontal, 1], normals[horizontal, 0])) % 360
    bins = int(round(360 / AZIMUTH_BIN_DEG))
    hist = np.bincount((azimuth / AZIMUTH_BIN_DEG).astype(int) % bins, area[horizontal], bins)
    half = int(round(tol / AZIMUTH_BIN_DEG))
    window = sum(np.roll(hist, k) for k in range(-half, half + 1))
    min_area = options.min_plane_area_frac * area.sum()
    directions, free = [], np.ones(bins, bool)
    while True:
        peak = int(np.argmax(np.where(free, window, -1)))
        if not free[peak] or window[peak] < min_area:
            break
        free[(peak + np.arange(-2 * half, 2 * half + 1)) % bins] = False
        offset = (azimuth - (peak + 0.5) * AZIMUTH_BIN_DEG + 180) % 360 - 180
        near = np.abs(offset) < tol / 2
        if area[horizontal][near].sum() <= 0:
            continue
        angle = (peak + 0.5) * AZIMUTH_BIN_DEG + np.average(offset[near], weights=area[horizontal][near])
        axis = 90 * round(angle / 90)
        if abs(angle - axis) <= options.manhattan_snap_deg:
            angle = axis
        rad = np.radians(angle)
        directions.append(np.array([np.cos(rad), np.sin(rad), 0.0]))
    return directions


def _offset_peaks(offsets: np.ndarray, area: np.ndarray, tau: float, min_area: float) -> list[float]:
    """Offsets with at least `min_area` within ±tau: peaks of the area-weighted histogram (bin tau/2), at least 2 tau
    apart, each refined to the area-weighted mean of its window."""
    width = tau / 2
    lo = offsets.min()
    idx = ((offsets - lo) / width).astype(int)
    hist = np.bincount(idx, area)
    window = np.convolve(hist, np.ones(5))[2 : 2 + len(hist)]  # ±2 bins = ±tau, same length as hist
    peaks, free = [], np.ones(len(hist), bool)
    while True:
        peak = int(np.argmax(np.where(free, window, -1)))
        if not free[peak] or window[peak] < min_area:
            break
        free[max(0, peak - 4) : peak + 5] = False
        centre = lo + (peak + 0.5) * width
        near = np.abs(offsets - centre) < tau
        if area[near].sum() <= 0:
            continue
        offset = float(np.average(offsets[near], weights=area[near]))
        if all(abs(offset - other) >= 2 * tau for other in peaks):
            peaks.append(offset)
    return peaks


def _fit(points: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, float]:
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
    if normal @ direction < 0:
        normal = -normal
    return normal, float(normal @ centre)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between the lines of two normals (opposite normals are parallel planes: 0)."""
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def detect_planes(
    mesh: trimesh.Trimesh, tau: float, options: PlaneOptions | None = None, weights: np.ndarray | None = None
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Planes of `mesh` as dicts (`normal`, `d`, `direction`, `axis_angle_deg`, `label`, `area`) and the plane index
    of every face (-1: none). Candidates are the offset peaks along ±Z and the wall directions; each keeps its faces
    with normal within `normal_tol_deg` and offset within tau, split into face-connected regions (small ones
    dropped), refit by SVD and snapped onto the direction when within `manhattan_snap_deg`. `weights` (one per face)
    scale the area each face counts for in the detection; the reported plane areas are the real ones."""
    options = options or PlaneOptions()
    normals, centres = mesh.face_normals, mesh.triangles_center
    real_area = mesh.area_faces
    area = real_area * (np.ones(len(real_area)) if weights is None else weights)
    total = float(area.sum())
    min_area = options.min_plane_area_frac * total
    cos_tol = np.cos(np.radians(options.normal_tol_deg))
    directions = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    directions += _wall_directions(normals, area, options)

    candidate = np.full(len(area), -1)
    peaks = []  # (direction, offset)
    for direction in directions:
        aligned = np.flatnonzero((normals @ direction > cos_tol) & (candidate < 0))
        if not len(aligned):
            continue
        offsets = centres[aligned] @ direction
        for offset in _offset_peaks(offsets, area[aligned], tau, min_area):
            members = aligned[(np.abs(offsets - offset) < tau) & (candidate[aligned] < 0)]
            candidate[members] = len(peaks)
            peaks.append((direction, offset))

    # face-connected regions of each candidate, in one pass over the face adjacency
    adjacency = mesh.face_adjacency
    same = (candidate[adjacency[:, 0]] >= 0) & (candidate[adjacency[:, 0]] == candidate[adjacency[:, 1]])
    pairs = adjacency[same]
    graph = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(area),) * 2)
    region = connected_components(graph, directed=False)[1]
    region_area = np.bincount(region, area)
    keep = (candidate >= 0) & (region_area[region] >= MIN_REGION_SHARE * min_area)
    candidate = np.where(keep, candidate, -1)

    planes, labels = [], np.full(len(area), -1)
    for index, (direction, _) in enumerate(peaks):
        faces = np.flatnonzero(candidate == index)
        if area[faces].sum() < min_area:
            continue
        normal, d = _fit(mesh.vertices[np.unique(mesh.faces[faces])], direction)
        angle = _angle_deg(normal, direction)
        if angle <= options.manhattan_snap_deg:
            normal = direction
            d = float(np.average(centres[faces] @ direction, weights=area[faces]))
        planes.append({"normal": normal, "d": d, "direction": direction, "axis_angle_deg": angle, "faces": faces})

    planes = _merge(planes, mesh, area, tau, options)
    for i, plane in enumerate(planes):
        labels[plane["faces"]] = i
        plane["area"] = float(real_area[plane["faces"]].sum())
    _label(planes, tau)
    return planes, labels


def _merge(planes: list[dict], mesh: trimesh.Trimesh, area: np.ndarray, tau: float, options: PlaneOptions) -> list:
    """Planes closer than `merge_angle_deg` in angle and tau in offset become one (faces joined, refit)."""
    merged = True
    while merged:
        merged = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                if _angle_deg(a["normal"], b["normal"]) < options.merge_angle_deg and abs(a["d"] - b["d"]) < tau:
                    faces = np.concatenate([a["faces"], b["faces"]])
                    a["faces"] = faces
                    if a["axis_angle_deg"] <= options.manhattan_snap_deg:
                        a["d"] = float(np.average(mesh.triangles_center[faces] @ a["normal"], weights=area[faces]))
                    else:
                        a["normal"], a["d"] = _fit(mesh.vertices[np.unique(mesh.faces[faces])], a["direction"])
                    del planes[j]
                    merged = True
                    break
            if merged:
                break
    return planes


def _label(planes: list[dict], tau: float) -> None:
    """floor: the largest upward plane within 2 tau of z = 0; ceiling: the highest downward plane with at least a
    quarter of the largest downward plane's area; wall: horizontal normal; other: the rest."""
    for plane in planes:
        plane["label"] = "wall" if abs(plane["normal"][2]) < np.sin(np.radians(5)) else "other"
    up = [p for p in planes if p["normal"][2] > 0.9 and abs(p["d"]) < 2 * tau]
    if up:
        max(up, key=lambda p: p["area"])["label"] = "floor"
    down = [p for p in planes if p["normal"][2] < -0.9]
    if down:
        largest = max(p["area"] for p in down)
        max((p for p in down if p["area"] >= 0.25 * largest), key=lambda p: -p["d"])["label"] = "ceiling"


def _project(x: np.ndarray, normals: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """The closest point to `x` on the intersection of the planes (rows of `normals`, offsets `ds`)."""
    residual = normals @ x - ds
    return x - normals.T @ np.linalg.solve(normals @ normals.T, residual)


def _snap(
    vertices: np.ndarray, vnormals: np.ndarray, mesh: trimesh.Trimesh, planes: list[dict], tau: float,
    options: PlaneOptions,
) -> tuple[np.ndarray, dict[str, Any]]:  # fmt: skip
    """New vertex positions and snap statistics. A vertex is a candidate of a plane when it is a vertex of the
    plane's region within the move limit of it, or within tau of the plane and ZONE_FACTOR tau of its region. One
    candidate: projected when the vertex is in the region or its normal is within `normal_tol_deg`; two or three
    non-parallel ones: moved onto their intersection line or point. Moves above `max_snap_move_factor` tau are
    rejected."""
    cos_tol = np.cos(np.radians(options.normal_tol_deg))
    max_move = options.max_snap_move_factor * tau
    candidates: list[list[int]] = [[] for _ in range(len(vertices))]
    in_region: list[set[int]] = [set() for _ in range(len(vertices))]
    for index, plane in enumerate(planes):
        dist_plane = np.abs(vertices @ plane["normal"] - plane["d"])
        region = np.unique(mesh.faces[plane["faces"]])
        for v in region[dist_plane[region] <= max_move]:
            in_region[v].add(index)
        near = np.flatnonzero(dist_plane < tau)
        dist = cKDTree(vertices[region]).query(vertices[near], distance_upper_bound=ZONE_FACTOR * tau)[0]
        for v in set(near[np.isfinite(dist)].tolist()) | {v for v in region if index in in_region[v]}:
            candidates[v].append(index)

    normals = np.array([p["normal"] for p in planes]).reshape(-1, 3)
    ds = np.array([p["d"] for p in planes])
    new = vertices.copy()
    kind = np.zeros(len(vertices), np.int8)  # 0 free, 1 plane, 2 edge, 3 corner, -1 rejected
    edge_pairs, corner_sets = set(), set()
    for v, cands in enumerate(candidates):
        if not cands:
            continue
        x = vertices[v]
        cands = sorted(cands, key=lambda i: abs(normals[i] @ x - ds[i]))
        chosen = [cands[0]]
        if options.straighten_edges:
            for i in cands[1:]:
                if all(_axis_angle_deg(normals[i], normals[j]) >= MIN_PAIR_ANGLE_DEG for j in chosen):
                    chosen.append(i)
                    if len(chosen) == 3:
                        break
        if len(chosen) == 3 and abs(np.linalg.det(normals[chosen])) < 0.1:
            chosen = chosen[:2]
        if len(chosen) == 1 and chosen[0] not in in_region[v] and vnormals[v] @ normals[chosen[0]] < cos_tol:
            continue
        target = _project(x, normals[chosen], ds[chosen])
        if np.linalg.norm(target - x) > max_move:
            kind[v] = -1
            continue
        new[v] = target
        kind[v] = len(chosen)
        if len(chosen) == 2:
            edge_pairs.add(tuple(sorted(chosen)))
        elif len(chosen) == 3:
            corner_sets.add(tuple(sorted(chosen)))
    return new, {"kind": kind, "edge_pairs": len(edge_pairs), "corners": len(corner_sets)}


def _unflip(
    faces: np.ndarray, before: np.ndarray, after: np.ndarray, moved: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Undo the moves of the vertices of faces whose normal turned around; returns the positions, the vertices
    still moved and the number of faces still flipped."""
    old = np.cross(before[faces[:, 1]] - before[faces[:, 0]], before[faces[:, 2]] - before[faces[:, 0]])
    for _ in range(FLIP_PASSES + 1):
        cur = np.cross(after[faces[:, 1]] - after[faces[:, 0]], after[faces[:, 2]] - after[faces[:, 0]])
        flipped = np.einsum("ij,ij->i", old, cur) < 0
        if not flipped.any():
            return after, moved, 0
        revert = np.unique(faces[flipped])
        revert = revert[moved[revert]]
        if not len(revert):
            break
        after[revert] = before[revert]
        moved[revert] = False
    return after, moved, int(flipped.sum())


def _taubin(mesh: trimesh.Trimesh, vertices: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Taubin smoothing (uniform weights) of the `free` vertices; the others stay where they are."""
    edges = mesh.edges_unique
    n = len(vertices)
    adj = sp.coo_matrix((np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(n, n)).tocsr()
    degree = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)
    x = vertices.copy()
    for _ in range(TAUBIN_STEPS):
        for factor in (TAUBIN_LAMBDA, TAUBIN_MU):
            delta = adj @ x / degree[:, None] - x
            x[free] += factor * delta[free]
    return x


def _rms(points: np.ndarray, normal: np.ndarray, d: float) -> float:
    return float(np.sqrt(np.mean((points @ normal - d) ** 2))) if len(points) else 0.0


def _patch_weights(mesh: trimesh.Trimesh, patch: Path | None, weight: float) -> np.ndarray | None:
    """Per-face weights: `weight` for the faces that are also faces of the mesh in `patch` (matched by centroid), else 1."""
    if patch is None or weight == 1.0 or not patch.is_file():
        return None
    other = read_mesh(patch)
    if not len(other.faces):
        return None
    dist = cKDTree(other.triangles_center).query(mesh.triangles_center)[0]
    return np.where(dist < 1e-6 * max(float(np.ptp(mesh.bounds, axis=0).max()), 1.0), weight, 1.0)


def _remove_bend(
    mesh: trimesh.Trimesh, vertices: np.ndarray, vnormals: np.ndarray, new: np.ndarray, kind: np.ndarray,
    planes: list[dict], tau: float, diag: float, options: PlaneOptions,
) -> int:  # fmt: skip
    """Move onto its plane every vertex facing the plane within `bend_max_tau` that the snap left alone because the
    surface is bent away from the plane: the bend is the offset to the plane smoothed over cells of `wave_cell_frac` of the diagonal
    (median per cell, then blurred); a vertex within `relief_tol_frac` of the bend follows it back to the plane, one
    further away is relief (a frame, a shelf) and stays, unless its piece of relief is smaller than
    `relief_min_frac`. Vertices of two planes (edges, corners) and moves above `bend_max_tau` are left. `new` and
    `kind` are updated in place; returns the vertices moved."""
    cell = options.wave_cell_frac * diag
    tol = options.relief_tol_frac * diag
    owners = np.zeros(len(vertices), int)  # structural planes a vertex belongs to (edges and corners: more than one)
    for plane in planes:
        if plane["label"] in ("floor", "wall", "ceiling"):
            owners[np.unique(mesh.faces[plane["faces"]])] += 1
    cos_tol = np.cos(np.radians(options.normal_tol_deg))
    moved = 0
    for plane in planes:
        if plane["label"] not in ("floor", "wall", "ceiling"):
            continue
        n = np.asarray(plane["normal"])
        offset_all = vertices @ n - plane["d"]
        region = np.flatnonzero((np.abs(offset_all) <= options.bend_max_tau * tau) & (vnormals @ n > cos_tol))
        offset = offset_all[region]
        near = np.ones(len(region), bool)
        if near.sum() < 10:
            continue
        helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uv = np.stack([vertices[region] @ u, vertices[region] @ v], axis=1)
        ij = np.floor((uv - uv[near].min(axis=0)) / cell).astype(int)
        shape = tuple(ij[near].max(axis=0) + 1)
        inside = near & (ij < np.array(shape)).all(axis=1) & (ij >= 0).all(axis=1)
        flat = np.ravel_multi_index(ij[inside].T, shape)
        grid, weight = np.zeros(shape), np.zeros(shape)
        medians = pd.Series(offset[inside]).groupby(flat).median()
        i, j = np.unravel_index(medians.index.to_numpy(), shape)
        grid[i, j], weight[i, j] = medians.to_numpy(), 1.0
        blur_w = ndimage.gaussian_filter(weight, 1.0)
        wave = ndimage.gaussian_filter(grid, 1.0) / np.maximum(blur_w, 1e-9)
        bend = np.zeros(len(region))
        bend[inside] = wave[ij[inside, 0], ij[inside, 1]]
        relief = inside & (np.abs(offset - bend) > tol)
        if relief.any():  # small pieces of relief are noise
            local = {int(g): i for i, g in enumerate(region)}
            rel_set = set(region[relief].tolist())
            edges = mesh.edges_unique
            pick = np.array([a in rel_set and b in rel_set for a, b in edges], bool)
            if pick.any():
                sub = edges[pick]
                graph = sp.coo_matrix((np.ones(len(sub)), (sub[:, 0], sub[:, 1])), shape=(len(vertices),) * 2)
                labels = connected_components(graph, directed=False)[1]
                members = region[relief]
                for label in np.unique(labels[members]):
                    group = members[labels[members] == label]
                    if np.ptp(vertices[group], axis=0).max() < options.relief_min_frac * diag:
                        relief[[local[int(g)] for g in group]] = False
        movable = inside & ~relief
        ids = region[movable]
        ids = ids[owners[ids] <= 1]
        if not len(ids):
            continue
        dist = vertices[ids] @ n - plane["d"]
        new[ids] = vertices[ids] - np.outer(dist, n)
        kind[ids] = 1
        moved += len(ids)
    return moved


def regularize(
    src: Path,
    out: Path,
    tau: float,
    options: PlaneOptions | None = None,
    patch: Path | None = None,
    diag: float | None = None,
) -> tuple[dict[str, Any], list[Plane]]:
    """Detect the planes of the mesh `src` (the faces of the mesh in `patch`, the gap fill, count
    `fill_patch_weight`), snap their vertices (straightening edges and corners), remove the bend of the walls when
    `bend_removal` (needs `diag`), Taubin-smooth the rest and write the PLY `out`; a copy of `src` when disabled.
    Returns the report and the planes."""
    options = options or PlaneOptions()
    if not options.enabled:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return {"enabled": False, "tau": tau}, []
    mesh = read_mesh(src)
    vertices = np.asarray(mesh.vertices, dtype=float)
    found, _ = detect_planes(mesh, tau, options, _patch_weights(mesh, patch, options.fill_patch_weight))
    new, snap = _snap(vertices, _unit(mesh.vertex_normals), mesh, found, tau, options)
    kind = snap["kind"]
    bent = _remove_bend(mesh, vertices, _unit(mesh.vertex_normals), new, kind, found, tau, diag, options) if options.bend_removal and diag else 0
    moved = kind > 0
    if options.smooth_nonplanar:
        boundary = np.zeros(len(vertices), bool)
        boundary[np.unique(mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)])] = True
        free = ~moved & ~boundary
        new = _taubin(mesh, new, free)
        moved |= free
    new, moved, flipped = _unflip(np.asarray(mesh.faces), vertices, new, moved)
    planar = (kind > 0) & moved
    reverted = int(((kind > 0) & ~moved).sum())

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=new, face_matrix=np.asarray(mesh.faces)))
    ms.compute_normal_per_vertex()
    save(ms, out)

    total = float(mesh.area_faces.sum())
    planes = []
    for i, p in enumerate(found):
        region = np.unique(mesh.faces[p["faces"]])
        planes.append(
            Plane(
                id=i,
                label=p["label"],
                normal=[float(c) for c in p["normal"]],
                d=p["d"],
                area=p["area"],
                area_frac=p["area"] / total,
                faces=len(p["faces"]),
                rms_before=_rms(vertices[region], p["normal"], p["d"]),
                rms_after=_rms(new[region], p["normal"], p["d"]),
                axis_angle_deg=p["axis_angle_deg"],
            )
        )
    move = np.linalg.norm(new - vertices, axis=1)[planar]
    walls = [p for p in planes if p.label == "wall" and p.area_frac >= WALL_RMS_MIN_AREA_FRAC]
    report = {
        "enabled": True,
        "tau": tau,
        "count": len(planes),
        "by_label": {label: sum(p.label == label for p in planes) for label in LABELS},
        "coverage": sum(p.area for p in planes) / total if total else 0.0,
        "snap": {
            "vertices": int(planar.sum()),
            "rejected": int((kind < 0).sum()),
            "reverted": reverted,
            "mean": float(move.mean()) if len(move) else 0.0,
            "p95": float(np.percentile(move, 95)) if len(move) else 0.0,
            "max": float(move.max()) if len(move) else 0.0,
        },
        "edges": {"pairs": snap["edge_pairs"], "vertices": int((planar & (kind == 2)).sum())},
        "corners": {"points": snap["corners"], "vertices": int((planar & (kind == 3)).sum())},
        "wall_rms_max": max((p.rms_after for p in walls), default=None),
        "flipped_faces": flipped,
        "bend_removed": bent,
    }
    return report, planes


def write_planes(path: Path, planes: list[Plane]) -> Path:
    return write_json([asdict(p) for p in planes], path)
