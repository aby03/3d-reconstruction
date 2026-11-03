"""Camera exports: COLMAP text models and a nerfstudio / instant-ngp `transforms.json` of the undistorted (PINHOLE)
cameras, with a reprojection check."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.io.tables import read_json, write_json

GL_FROM_CV = np.diag([1.0, -1.0, -1.0])  # camera axes: OpenCV (x right, y down, z forward) -> OpenGL (y up, z back)
CHECK_POINTS = 10  # sparse points re-projected through the exported cameras
MAX_REPROJ_PX = 1.0


def write_colmap_text(model_dir: Path, out: Path) -> dict[str, int]:
    """The COLMAP model in `model_dir` as a text model (cameras.txt, images.txt, points3D.txt, ...) in `out`."""
    rec = pycolmap.Reconstruction(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec.write_text(out)
    return {"cameras": rec.num_cameras(), "images": rec.num_reg_images(), "points": rec.num_points3D()}


def _intrinsics(camera: pycolmap.Camera) -> dict[str, Any]:
    if camera.model.name != "PINHOLE":
        raise ValueError(f"camera {camera.camera_id}: {camera.model.name}, expected PINHOLE (undistorted)")
    fx, fy, cx, cy = camera.params
    return {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": camera.width, "h": camera.height}


def transforms_json(undistorted_model: Path, images_dir: Path, out: Path) -> dict[str, Any]:
    """`out` (nerfstudio `transforms.json`) for the undistorted model `undistorted_model` whose images are in
    `images_dir`: PINHOLE intrinsics (top level for one camera, per frame otherwise), per frame `file_path`
    (relative to `out`'s folder) and `transform_matrix` = camera-to-world in W with OpenGL camera axes."""
    rec = pycolmap.Reconstruction(undistorted_model)
    cameras = {cid: _intrinsics(cam) for cid, cam in rec.cameras.items()}
    frames = []
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        world_from_cam = image.cam_from_world().inverse().matrix()  # 3x4
        matrix = np.eye(4)
        matrix[:3, :3] = world_from_cam[:, :3] @ GL_FROM_CV
        matrix[:3, 3] = world_from_cam[:, 3]
        frame = {
            "file_path": os.path.relpath(images_dir / image.name, out.parent),
            "transform_matrix": matrix.tolist(),
            "colmap_im_id": image.image_id,
        }
        if len(cameras) > 1:
            frame.update(cameras[image.camera_id])
        frames.append(frame)
    tree: dict[str, Any] = {"camera_model": "PINHOLE"}
    if len(cameras) == 1:
        tree.update(next(iter(cameras.values())))
    tree.update({"world_up": "+Z", "frames": frames})
    write_json(tree, out)
    return {"frames": len(frames), "cameras": len(cameras)}


def _project(frame: dict[str, Any], intrinsics: dict[str, Any], xyz: np.ndarray) -> np.ndarray:
    matrix = np.asarray(frame["transform_matrix"])
    world_from_cam = matrix[:3, :3] @ GL_FROM_CV  # back to OpenCV axes
    cam = world_from_cam.T @ (xyz - matrix[:3, 3])
    x, y = cam[:2] / cam[2]
    return np.array([intrinsics["fl_x"] * x + intrinsics["cx"], intrinsics["fl_y"] * y + intrinsics["cy"]])


def validate_transforms(path: Path, model: Path) -> dict[str, Any]:
    """Check `transforms.json` against the model it came from: every `file_path` exists, every registered image
    has a frame, and the CHECK_POINTS sparse points with at least 3 observations and the lowest reprojection error,
    projected through the exported cameras, land within MAX_REPROJ_PX of their observations. Raises ValueError."""
    tree = read_json(path)
    frames = {frame["colmap_im_id"]: frame for frame in tree["frames"]}
    if missing := [f["file_path"] for f in frames.values() if not (path.parent / f["file_path"]).is_file()]:
        raise ValueError(f"{path.name}: {len(missing)} image(s) missing, e.g. {missing[0]}")
    rec = pycolmap.Reconstruction(model)
    if set(frames) != set(rec.reg_image_ids()):
        raise ValueError(f"{path.name}: {len(frames)} frames for {rec.num_reg_images()} registered images")
    tracked = [(p.error, pid) for pid, p in rec.points3D.items() if p.track.length() >= 3]
    points = [rec.points3D[pid] for _, pid in sorted(tracked)]
    errors = []
    for point in points[:CHECK_POINTS]:
        for element in point.track.elements:
            frame = frames[element.image_id]
            intrinsics = frame if "fl_x" in frame else tree
            observed = rec.images[element.image_id].points2D[element.point2D_idx].xy
            errors.append(float(np.linalg.norm(_project(frame, intrinsics, point.xyz) - observed)))
    max_px = max(errors, default=0.0)
    if max_px > MAX_REPROJ_PX:
        raise ValueError(f"{path.name}: sparse points re-project {max_px:.2f} px from their observations")
    checked = min(CHECK_POINTS, len(points))
    return {"frames": len(frames), "points": checked, "observations": len(errors), "max_px": max_px}
