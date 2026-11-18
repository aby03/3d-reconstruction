"""Real-time viewer of a run's 2DGS splats (dense.backend 2dgs) in the browser, rendered on the local GPU with
gsplat's viser / nerfview viewer. Needs the gsplat env:

    pixi run -e gsplat view-splats --run outputs/<scene>/runs/<run_id> [--port 8080] [--mesh]
    pixi run -e gsplat view-splats --run RUN --check     # render one frame per mode offscreen and exit

Open http://localhost:<port>. Controls (panel "Splats"): render mode, resolution scale, depth colour range,
background, input camera frustums (click one to jump to it), jump to camera #, play the novel-view path, mesh toggle.
"""

import argparse
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from recon.dense.gs_render import novel_path, render
from recon.dense.gs_train import Splats, scaled_K
from recon.pipeline.stages import s05_dense, s06_mesh_post

MODES = ("RGB", "depth", "normal")
MAX_MESH_FACES = 300_000
MAX_DEPTH = 167.0  # viser's fixed-point depth range: pixels without a surface go behind every scene object
PATH_FPS = 20


def load(run: Path) -> tuple[Splats, list[dict], pycolmap.Reconstruction]:
    """The splats, the input cameras ({name, K, w2c, size}, in name order) and the undistorted model of a run."""
    dense = run / s05_dense.DIR_NAME
    ply = dense / s05_dense.GS_ARTIFACTS["splats"]
    if not ply.is_file():
        ply = run / "07_export" / "splats" / "splats.ply"
    rec = pycolmap.Reconstruction(dense / s05_dense.ARTIFACTS["undistorted_model"])
    cams = []
    for im in sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name):
        cam, w2c = rec.cameras[im.camera_id], np.eye(4, dtype=np.float32)
        w2c[:3] = im.cam_from_world().matrix()
        cams.append({"name": im.name, "K": scaled_K(cam, cam.width, cam.height), "w2c": w2c,
                     "size": (cam.width, cam.height)})
    s = Splats.load_ply(ply)  # on the GPU once: render() then moves nothing per frame
    return Splats(*(t.to("cuda") for t in (s.means, s.quats, s.scales, s.opacities, s.sh))), cams, rec


def load_mesh(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """`mesh_final_geometry.ply`, decimated to at most MAX_MESH_FACES faces: (vertices, faces)."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(run / s06_mesh_post.DIR_NAME / s06_mesh_post.ARTIFACTS["mesh_final_geometry"]))
    if len(mesh.triangles) > MAX_MESH_FACES:
        mesh = mesh.simplify_quadric_decimation(MAX_MESH_FACES)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)


def frame(splats: Splats, K, w2c, size, mode: str, depth_range=(0.0, 0.0), bg=(0, 0, 0)):
    """One view as (uint8 RGB image, float32 depth for viser's compositing) in `mode`; depth_range (0, 0) = auto."""
    r = {k: v.float().cpu().numpy() for k, v in render(splats, K, w2c, size).items()}
    a, d = r["alpha"][..., None], r["depth"]
    if mode == "depth":
        hit = d[r["alpha"] > 0.5]
        lo, hi = depth_range if depth_range[1] > depth_range[0] else np.percentile(hit, [1, 99]) if hit.size else (0, 1)
        t = (np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(t, cv2.COLORMAP_TURBO)[..., ::-1] / 255 * a
    else:
        img = r["rgb"] if mode == "RGB" else (r["normal"] * 0.5 + 0.5) * a
    img = img + (1 - a) * np.asarray(bg, np.float32) / 255
    depth = np.where(r["alpha"] > 0.5, d, MAX_DEPTH).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), depth


def check(splats: Splats, cams: list[dict], mesh) -> None:
    """Render the first input camera in every mode; fail on an empty or non-finite frame."""
    c = cams[0]
    for mode in MODES:
        t0 = time.perf_counter()
        img, depth = frame(splats, c["K"], c["w2c"], c["size"], mode)
        assert img.shape == (c["size"][1], c["size"][0], 3) and img.any() and np.isfinite(depth).all(), mode
        print(f"{mode}: {img.shape[1]}x{img.shape[0]} in {time.perf_counter() - t0:.3f} s")
    if mesh is not None:
        print(f"mesh: {len(mesh[1])} faces")
    print(f"ok: {len(splats)} splats, {len(cams)} cameras")


def serve(splats: Splats, cams: list[dict], rec: pycolmap.Reconstruction, mesh, port: int, novel: int) -> None:
    import nerfview
    import viser
    import viser.transforms as vt

    server = viser.ViserServer(port=port, verbose=False)
    gui = {}

    def render_fn(camera, tab):
        w, h = (max(1, round(s * gui["scale"].value)) for s in (tab.viewer_width, tab.viewer_height))
        return frame(splats, camera.get_K((w, h)), np.linalg.inv(camera.c2w), (w, h), gui["mode"].value,
                     (gui["near"].value, gui["far"].value), gui["bg"].value)

    viewer = nerfview.Viewer(server, render_fn, output_dir=Path(tempfile.gettempdir()) / "view_splats",
                             mode="rendering")
    up = -np.mean([c["w2c"][1, :3] for c in cams], 0)  # camera y points down
    server.scene.set_up_direction(up / np.linalg.norm(up))

    def pose(w2c, K, size):
        R = w2c[:3, :3].T
        return vt.SO3.from_matrix(R).wxyz, -R @ w2c[:3, 3], 2 * np.arctan(size[1] / 2 / K[1, 1])

    def go(w2c, K, size, clients=None):
        wxyz, pos, fov = pose(w2c, K, size)
        for client in (clients or server.get_clients()).values():
            with client.atomic():
                client.camera.wxyz, client.camera.position, client.camera.fov = wxyz, pos, fov

    with server.gui.add_folder("Splats"):
        gui["mode"] = server.gui.add_dropdown("Mode", MODES, initial_value="RGB")
        gui["scale"] = server.gui.add_slider("Resolution scale", 0.1, 1.0, 0.05, 1.0)
        gui["near"] = server.gui.add_number("Depth near", 0.0, step=0.01, hint="depth colours; near = far: auto")
        gui["far"] = server.gui.add_number("Depth far", 0.0, step=0.01)
        gui["bg"] = server.gui.add_rgb("Background", (0, 0, 0))
        frustums_on = server.gui.add_checkbox("Camera frustums", True)
        index = server.gui.add_slider("Camera #", 0, max(len(cams) - 1, 1), 1, 0)
        jump, play = server.gui.add_button("Go to camera"), server.gui.add_button("Play novel path")
        mesh_on = server.gui.add_checkbox("Mesh", True) if mesh is not None else None
    go_cam = lambda c, clients=None: go(c["w2c"], c["K"], c["size"], clients)  # noqa: E731
    for handle in gui.values():
        handle.on_update(viewer.rerender)

    scale = 0.05 * np.linalg.norm(np.ptp([pose(c["w2c"], c["K"], c["size"])[1] for c in cams], 0))
    frustums = []
    for i, c in enumerate(cams):
        wxyz, pos, fov = pose(c["w2c"], c["K"], c["size"])
        f = server.scene.add_camera_frustum(f"/cams/{i:04d}", fov, c["size"][0] / c["size"][1], max(scale, 1e-3),
                                            color=(255, 160, 40), wxyz=wxyz, position=pos)
        f.on_click(lambda _, c=c: go_cam(c))
        frustums.append(f)

    @frustums_on.on_update
    def _(_):
        for f in frustums:
            f.visible = frustums_on.value

    jump.on_click(lambda _: go_cam(cams[int(index.value)]))
    path, size = novel_path(rec, novel, max(cams[0]["size"]))

    def run_path():
        for K, w2c in path:
            go(w2c, K, size)
            time.sleep(1 / PATH_FPS)

    play.on_click(lambda _: threading.Thread(target=run_path, daemon=True).start())
    if mesh is not None:
        handle = server.scene.add_mesh_simple("/mesh", *mesh, color=(180, 180, 200), flat_shading=True)
        mesh_on.on_update(lambda _: setattr(handle, "visible", mesh_on.value))
    server.on_client_connect(lambda client: go_cam(cams[0], {0: client}))
    print(f"viewer at http://localhost:{port} ({len(splats)} splats, {len(cams)} cameras); Ctrl+C to stop", flush=True)
    while True:
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory of a dense.backend=2dgs run")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mesh", action="store_true", help="also show mesh_final_geometry (toggle in the panel)")
    parser.add_argument("--novel", type=int, default=240, help="frames of the novel-view path")
    parser.add_argument("--check", action="store_true", help="render one frame per mode offscreen and exit")
    args = parser.parse_args()
    splats, cams, rec = load(args.run)
    mesh = load_mesh(args.run) if args.mesh else None
    if args.check:
        check(splats, cams, mesh)
    else:
        serve(splats, cams, rec, mesh, args.port, args.novel)


if __name__ == "__main__":
    main()
