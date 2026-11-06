"""Texture fixes on OpenMVS's atlases. TextureMesh gives every face no view saw one shared UV point on an empty
(magenta) texel; each such face gets a colour cell in the atlas's empty space instead, coloured like the dense
points around the face (faces of the same quantized colour share a cell). Empty texels inside textured faces' UV triangles get the same kind of colour. The texels of
the faces' UV triangles then bleed into the empty space around them (gutters, so bilinear filtering and mipmaps don't
pull in the empty colour), the rest of the empty space gets the mean colour, and the MTL is made opaque."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial import cKDTree

from recon.io.ply import read_openmvs_points
from recon.mesh.texturing import TEXTURED_OBJ, _is_sentinel, opaque_mtl, read_mtl_maps, read_obj_faces, read_obj_uv

SUBPIXEL_BITS = 4  # cv2.fillPoly fixed-point precision
CELL_PX = 16  # colour cells of untextured faces: one JPEG chroma block (4:2:0), so the colour survives encoding

FaceColors = Callable[[np.ndarray], np.ndarray]  # face indices -> (n, 3) colours in the image's channel order


@dataclass(slots=True)
class TextureFixOptions:
    fill_untextured: bool = True
    gutter_px: int = 8
    jpeg_quality: int = 95  # PNG would be ~4x the bytes
    fill_k: int = 8  # dense points averaged per face
    color_step: int = 8  # untextured faces whose colours round to the same multiple of this share a colour cell


def _pixel_triangles(uv: np.ndarray, faces: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(F, 3, 2) fixed-point pixel coordinates of the faces' UV triangles (texel centres at integer + 0.5; OBJ v
    points up)."""
    w, h = size
    tri = uv[faces]
    xy = np.stack([tri[..., 0] * w - 0.5, (1.0 - tri[..., 1]) * h - 0.5], axis=-1)
    return np.round(xy * (1 << SUBPIXEL_BITS)).astype(np.int32)


def _overlapping(tris: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices of the triangles whose pixel bounding box contains a texel of `mask` (summed-area table)."""
    h, w = mask.shape
    table = cv2.integral(mask.astype(np.uint8))
    lo = np.floor(tris.min(axis=1) / (1 << SUBPIXEL_BITS)).astype(int)
    hi = np.ceil(tris.max(axis=1) / (1 << SUBPIXEL_BITS)).astype(int) + 1
    x0, y0 = np.clip(lo[:, 0], 0, w), np.clip(lo[:, 1], 0, h)
    x1, y1 = np.clip(hi[:, 0], 0, w), np.clip(hi[:, 1], 0, h)
    count = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    return np.flatnonzero(count > 0)


def _box3(a: np.ndarray) -> np.ndarray:
    return cv2.boxFilter(a, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)


def _dilate(image: np.ndarray, used: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow the used texels `steps` texels into the unused ones, each new texel the mean of its used 3x3
    neighbours (one channel at a time: an 8192² float image is 268 MB)."""
    image, used = image.copy(), used.copy()
    for _ in range(steps):
        weight = used.astype(np.float32)
        count = _box3(weight)
        ring = ~used & (count > 0)
        if not ring.any():
            break
        for c in range(image.shape[2]):
            total = _box3(image[..., c].astype(np.float32) * weight)
            image[..., c][ring] = np.round(total[ring] / count[ring]).astype(np.uint8)
        used |= ring
    return image, used


def untextured_faces(image: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces without UVs or whose UV-triangle centroid samples the empty colour (as `atlas_stats` counts them)."""
    h, w = image.shape[:2]
    missing = (faces < 0).any(axis=1)
    centroid = uv[np.where(faces < 0, 0, faces)].mean(axis=1)
    x = np.clip((centroid[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1.0 - centroid[:, 1]) * h).astype(int), 0, h - 1)
    return missing | _is_sentinel(image)[y, x]


def _free_cells(free: np.ndarray) -> np.ndarray:
    """(N, 2) top-left pixel of every CELL_PX-aligned cell lying entirely in `free`."""
    h, w = free.shape
    rows, cols = h // CELL_PX, w // CELL_PX
    blocks = free[: rows * CELL_PX, : cols * CELL_PX].reshape(rows, CELL_PX, cols, CELL_PX).all(axis=(1, 3))
    by, bx = np.nonzero(blocks)
    return np.stack([bx, by], axis=1) * CELL_PX


def fix_atlas(
    image: np.ndarray, uv: np.ndarray, faces: np.ndarray, colors: FaceColors | None, options: TextureFixOptions
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Fix one atlas (H, W, 3) whose faces have the UV indices `faces` (F, 3) into `uv`. `colors` gives the fill
    colour of faces (None: nothing is filled). Returns the image, the new UV (u, v) of each untextured face that got
    a colour cell, and `{filled_faces (untextured faces given a colour cell), colour_cells, filled_texels (empty texels filled inside
    textured faces), unfilled_faces (untextured faces left: no free cell), sentinel_in_faces}`."""
    h, w = image.shape[:2]
    image = image.copy()
    untextured = untextured_faces(image, uv, faces)
    textured = np.flatnonzero(~untextured)
    tris = _pixel_triangles(uv, faces[textured], (w, h))
    covered = np.zeros((h, w), np.uint8)
    cv2.fillPoly(covered, list(tris), 1, lineType=cv2.LINE_8, shift=SUBPIXEL_BITS)
    covered = covered.astype(bool)
    holes = covered & _is_sentinel(image)

    filled_texels = 0
    if colors is not None and holes.any():
        owner = np.full((h, w), -1, np.int32)
        for i in _overlapping(tris, holes):
            cv2.fillPoly(owner, [tris[i]], int(i), lineType=cv2.LINE_8, shift=SUBPIXEL_BITS)
        target = holes & (owner >= 0)
        local, inverse = np.unique(owner[target], return_inverse=True)
        image[target] = np.clip(np.round(colors(textured[local])[inverse]), 0, 255).astype(np.uint8)
        filled_texels = int(target.sum())

    # only texels inside face triangles are kept: OpenMVS's patch margins carry a pinkish JPEG fringe of the empty
    # colour, so the gutters are grown again from the triangles
    used = covered
    image, grown = _dilate(image, used, options.gutter_px)

    cells: dict[int, np.ndarray] = {}
    palette_size = 0
    pending = np.flatnonzero(untextured)
    if colors is not None and len(pending):
        # faces of the same quantized colour share one cell; the step doubles until the palette fits the free space
        free = _free_cells(~grown)
        rgb = np.clip(np.round(colors(pending)), 0, 255).astype(int)
        step = max(1, options.color_step)
        while True:
            quantized = np.minimum((rgb // step) * step + step // 2, 255)
            palette, which = np.unique(quantized, axis=0, return_inverse=True)
            which = which.reshape(-1)
            if len(palette) <= len(free) or step >= 128:
                break
            step *= 2
        palette_size = min(len(palette), len(free))
        for (x, y), color in zip(free[:palette_size], palette[:palette_size].astype(np.uint8)):
            image[y : y + CELL_PX, x : x + CELL_PX] = color
            grown[y : y + CELL_PX, x : x + CELL_PX] = True
        centre = (free[:palette_size] + CELL_PX / 2) / [w, h]
        cells = {
            int(f): np.array([centre[k, 0], 1.0 - centre[k, 1]]) for f, k in zip(pending, which) if k < palette_size
        }
    if (~grown).any():
        image[~grown] = np.round(image[used].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    stats = {
        "filled_faces": len(cells),
        "colour_cells": palette_size,
        "filled_texels": filled_texels,
        "unfilled_faces": int(untextured.sum()) - len(cells),
        "sentinel_in_faces": int((covered & _is_sentinel(image)).sum()),
    }
    return image, cells, stats


def _write_obj(src: Path, out: Path, num_uv: int, cells: dict[str, dict[int, np.ndarray]]) -> None:
    """Copy the OBJ `src` to `out`, with one new texture coordinate per cell (after the existing ones) and the
    faces that got a cell pointing at it; everything else as in `src`."""
    new_uv, index = [], {}
    for name, per_face in cells.items():
        for face, coord in per_face.items():
            index[(name, face)] = num_uv + len(new_uv) + 1
            new_uv.append(coord)
    with src.open() as f_in, out.open("w") as f_out:
        current, counts, seen_vt = None, {}, False
        for line in f_in:
            if line.startswith("vt "):
                seen_vt = True
            elif seen_vt and new_uv:  # right after the last existing texture coordinate
                f_out.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in new_uv)
                new_uv = []
            if line.startswith("usemtl "):
                current = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                face = counts.get(current, 0)
                counts[current] = face + 1
                if (current, face) in index:
                    vt = index[(current, face)]
                    corners = [c.split("/") for c in line.split()[1:]]
                    line = "f " + " ".join("/".join([c[0], str(vt), *c[2:]]) for c in corners) + "\n"
            f_out.write(line)


def fix_textures(
    textured_dir: Path, out_dir: Path, dense_points: Path, options: TextureFixOptions | None = None
) -> dict[str, Any]:
    """Copy the textured mesh in `textured_dir` (OpenMVS's OBJ + MTL + JPEG atlases) to `out_dir` with fixed
    atlases (same names and sizes, JPEG `jpeg_quality`) and an opaque MTL. Faces and vertices are unchanged, and
    so are the texture coordinates of every textured face; untextured faces point at their colour cell. Colours
    come from the dense points of `dense_points` (`05_dense/dense_points.ply`) around each face."""
    options = options or TextureFixOptions()
    obj = textured_dir / TEXTURED_OBJ
    mtllib, uv, per_material, _ = read_obj_uv(obj)
    maps = read_mtl_maps(textured_dir / mtllib)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / mtllib).write_text(opaque_mtl((textured_dir / mtllib).read_text()))

    if options.fill_untextured:
        vertices, positions = read_obj_faces(obj)
        xyz, _, rgb, _ = read_openmvs_points(dense_points)
        tree = cKDTree(xyz)
        k = min(options.fill_k, len(xyz))

    def colors_for(name: str) -> FaceColors | None:
        if not options.fill_untextured:
            return None

        def colors(face_idx: np.ndarray) -> np.ndarray:
            centroids = vertices[positions[name][face_idx]].mean(axis=1)
            idx = tree.query(centroids, k=k)[1].reshape(len(centroids), k)
            return rgb[idx][..., ::-1].astype(float).mean(axis=1)  # BGR like the image

        return colors

    totals = {"filled_faces": 0, "colour_cells": 0, "filled_texels": 0, "unfilled_faces": 0, "sentinel_in_faces": 0}
    cells = {}
    for name, idx in per_material.items():
        image = cv2.imread(str(textured_dir / maps[name]))
        fixed, cells[name], stats = fix_atlas(image, uv, idx, colors_for(name), options)
        cv2.imwrite(str(out_dir / maps[name]), fixed, [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality])
        for key in totals:
            totals[key] += stats[key]
    _write_obj(obj, out_dir / TEXTURED_OBJ, len(uv), cells)
    return {**totals, "gutter_px": options.gutter_px, "jpeg_quality": options.jpeg_quality}
