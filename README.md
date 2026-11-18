# 3D room reconstruction

Turns a phone video with IMU data (Open Camera Sensors capture) into a textured, plane-regularized room mesh. SfM with COLMAP/GLOMAP, then one of two dense backends: OpenMVS (CUDA) with monocular-depth gap filling, or 2D Gaussian splatting (gsplat), which fills far more of the room and also gives photo-real novel views. Texturing with OpenMVS and an HTML quality report for both.

A novel view of the hotel1 reconstruction, from one viewpoint high in a room corner where no input camera was:

![Novel view of hotel1: OpenMVS textured mesh, 2DGS splat render, SfM points, camera path](results/hotel1/hero.png)

Top: the OpenMVS textured mesh (white: no surface) and the 2DGS splat render. Bottom: the SfM sparse points from the same viewpoint, and the same with the camera path of the walk (blue start to red end, arrows: viewing direction of every 12th camera).

The example capture `data/hotel1` (a hotel room: one 3146-frame phone video, 53 sharper reference stills and the IMU streams) ships in Git LFS; the results below are from it. The figures are rendered from the run's outputs by the report stage for every dataset (`08_report/figures/`); `scripts/make_readme_figures.py --run <run dir>` copies them to `results/`.

## Setup

Linux x86_64, NVIDIA GPU with a CUDA 12.9+ driver (developed on a GTX 1660 Ti), `git`, `git-lfs`.

```bash
wget -qO- https://pixi.sh/install.sh | sh                       # 1. pixi, then restart the shell
git lfs install                                                 # 2. once (sudo apt install git-lfs)
git clone --recurse-submodules https://github.com/aby03/3d-reconstruction.git && cd 3d-reconstruction
pixi install                                                    # 3. main environment
pixi install -e build && pixi run -e build openmvs              # 4. OpenMVS (develop) with CUDA (minutes, once)
pixi run recon doctor                                           # 5. GPU, CUDA paths, tool versions
pixi install -e gsplat                                          # 6. optional: 2DGS backend (gsplat compiles its CUDA code on first use, minutes)
```

Already cloned without submodules or LFS: `git submodule update --init --recursive`, `git lfs pull`. Step 4 builds two small dependencies from source (`halfmesh`, `tinyply`), applies `patches/openmvs-no-sfm.patch` (skips OpenMVS's own SfM library and its two apps, which we do not use and which need packages conda-forge lacks) and builds the `develop` branch of OpenMVS: it contains the TextureMesh fix for `--resolution-level` above 0 that made the 2.1 build hang.

## Run

```bash
pixi run recon run --input data/hotel1                 # every stage, dense backend OpenMVS (default)
pixi run -e gsplat recon run --input data/hotel1 --set dense.backend=2dgs   # every stage, dense backend 2DGS
pixi run recon run --input data/hotel1 --to keyframes  # stop after a stage
pixi run recon run --input data/hotel1 --resume        # continue the latest run, skipping up-to-date stages
pixi run recon compare RUN_A RUN_B --scene hotel1      # metrics of two runs side by side
pixi run recon list-stages
```

Switch an existing run to 2DGS, reusing its SfM and alignment, or continue an interrupted 2DGS training (`dense.gs.max_minutes` stops it after that long; the same command continues from `05_dense/gs/ckpt.pt`):

```bash
pixi run -e gsplat recon run --input data/hotel1 --resume RUN_ID --from dense --force dense --set dense.backend=2dgs [--set dense.gs.max_minutes=30]
```

Per-sequence stages write to `outputs/<scene>/sequences/<id>/<NN_stage>/`, scene stages to `outputs/<scene>/runs/<run_id>/<NN_stage>/`. A stage whose inputs and config are unchanged is skipped.

| Option | Meaning |
|---|---|
| `--from/--to/--only STAGE` | choose the stages to run |
| `--force STAGE` | re-run a stage and everything after it |
| `--resume [RUN_ID]` | continue in a run dir (no id: the latest) |
| `--set KEY=VALUE` | override a config key, e.g. `--set keyframes.stills=false` |
| `--config FILE` | YAML layered over `configs/default.yaml` |
| `--hardware PRESET` | `cuda_6gb`, `cuda_large`, `cpu` (default: picked by GPU memory) |

`configs/default.yaml` lists every key with its default; unknown keys are errors.

## Collect your own data

The pipeline reads captures made with the Android app [Open Camera Sensors](https://f-droid.org/en/packages/com.opencamera_sensors.app/) (F-Droid). It is the only capture source supported at the moment. The app records a video together with the raw accelerometer, gyroscope and magnetometer streams and a timestamp per video frame, and can save sharper reference JPEGs while recording.

1. Install the app from F-Droid, and in its settings turn on recording of all three sensors: the gyroscope (pair filter, rotation priors, camera-IMU check), the accelerometer (gravity direction that levels the model, motion check of the path) and the magnetometer (heading check of the path). A missing stream is a warning and the checks that need it are skipped.
2. Record one continuous video while walking through the room. Reference JPEGs saved during the recording are used as a second, sharper camera (`keyframes.stills`; turn off with `--set keyframes.stills=false`).
3. Copy the files from the phone into one folder, keeping the layout the app writes (`<stamp>` is `YYYYMMDD_HHMMSS`):

```
my_room/
  VID_<stamp>.mp4
  <stamp>/
    VID_<stamp>_imu_timestamps.csv      one timestamp (ns) per video frame
    VID_<stamp>accel.csv  VID_<stamp>gyro.csv  VID_<stamp>magnetic.csv      x,y,z,t_ns without header
    <t_ns>.jpg                          reference stills (optional)
```

4. Run it: `pixi run recon run --input my_room --scene my_room` (`data/hotel1` is a complete example).

Notes from the two captures used so far (`hotel1`, `room2`: about one minute each, about 370 images):

- Only captures of about a minute have been tested; longer ones have not been run.
- Walk slowly and keep the room in view. Textureless walls, windows and mirrors reconstruct poorly (gap filling patches the missing wall surface from monocular depth).
- Turning on the spot for a long time at the start of `hotel1` coincided with the global mapper cutting the camera path into pieces; the `sfm` stage detects this and re-maps with the incremental mapper, but a path that keeps moving is the safer capture.
- The loader checks that the frame timestamps match the video and that the reference JPEGs match the decoded frames (`--no-verify-refs` skips the second check). If the camera and IMU clocks are offset, `--time-offset-ns` shifts the frame times.

## Pipeline

| # | Stage | What it does | Main outputs |
|---|---|---|---|
| 1 | `ingest` | loads video, timestamps and IMU, checks frame timing, syncs sensors to frames, estimates gravity | `frames.csv`, sync report |
| 2 | `keyframes` | scores blur/exposure/motion, picks keyframes (adaptive to image motion and gyro); adds the reference stills as camera `cam1` | `images/cam0`, `images/cam1`, IMU priors |
| 3 | `sfm` | SIFT features (ALIKED, DISK, SuperPoint selectable), exhaustive or time-neighbour and loop matching, COLMAP global mapper with gravity priors and a drift-aware gyro pair filter, then refinement; checks that the camera path is continuous in time (jumps, isolated cameras, gyro and accelerometer windows, magnetometer) and re-maps with the incremental mapper when the path comes out in pieces | `sparse/refined`, `sfm_report.json` |
| 4 | `align` | world frame: Z up, floor at z = 0, X/Y along the walls; checks camera–IMU calibration | `sparse_aligned`, `transform.json` |
| 5 | `dense` | `openmvs`: undistort, OpenMVS densify (fusion filter 1), mesh and refine, outlier filter, Depth-Anything-V2 mono depth fitted to the mesh depth, TSDF fusion. `2dgs`: undistort, mono depth, train 2D Gaussians (gsplat, mono-depth prior), TSDF fusion of their rendered depth, held-out PSNR, novel-view video | `dense_points.ply`, `mesh_refined.ply`, `mesh_tsdf.ply`; 2dgs also `gs/splats.ply`, `gs/novel.mp4` |
| 6 | `mesh_post` | cleanup, gap fill from mono depth, hole closing, view-consistency check (OpenMVS only), plane snapping, decimation to 2M faces, texturing with seam leveling, texture fixes, QA | `mesh_final_geometry.ply`, `textured_fixed/` |
| 7 | `export` | GLB (Y up), OBJ, PLY, STL, point clouds, cameras (COLMAP, nerfstudio `transforms.json`), `metadata.json`; every file is reloaded and checked | `07_export/` |
| 8 | `report` | thresholds on every stage, hole share of the raw and final mesh, one HTML page | `08_report/index.html` |

Each stage also runs alone (`pixi run recon <stage> --input data/hotel1`) and writes `stage_manifest.json` plus a `reports/` folder with its metrics and plots.

Design points worth knowing:

- **Stills as a second camera.** Reference stills have about 3x the sharpness of the video frames and join SfM with their own intrinsics (`keyframes.stills`).
- **Gap filling.** Textureless walls leave the dense cloud empty. Patches of the mono-depth TSDF mesh are added where the OpenMVS mesh has no surface (`mesh_post.fill.*`); an optional weld joins them to the mesh but costs accuracy, so it is off.
- **Dense step reuse.** The dense stage saves its OpenMVS steps (undistort, densify, filter, mesh, refine, preview) with a key of their options and the aligned model. Re-running the stage with other depth or fusion settings (`--resume RUN --from dense`) skips them (hotel1: about 100 min saved); other `dense.*` OpenMVS options or a new alignment rerun everything. Runs made before this can be adopted with `scripts/adopt_dense_chain.py`. To keep the earlier run intact, copy it first with `cp -al`.
- **Selectable SfM features.** `sfm.features.type` is `sift` (default), `aliked` (COLMAP's ALIKED with LightGlue; the ONNX models are downloaded once into `~/.cache/colmap/`), or `disk` / `superpoint` (LightGlue; run with `pixi run -e lightglue recon ...`). `sfm.features.matcher` is `bruteforce` (default) or `lightglue` (COLMAP's, for sift / aliked; keypoints are then capped at 4096). Loop detection through the vocab tree (COLMAP downloads it once) is SIFT-only and runs with the `time` matching strategy.
- **Optional learned depth.** `dense.depth_source=simplerecon` replaces the mono depth by SimpleRecon (multi-view, metric; non-commercial licence). Install once with `pixi install -e simplerecon && pixi run -e simplerecon fetch-simplerecon`, then run with `pixi run -e simplerecon recon ... --set dense.depth_source=simplerecon`. The pose scale (model units per metre) is searched on a sample of views, starting from the SfM accelerometer scale or the median SfM depth, and scored against the SfM points; if nothing agrees, or SimpleRecon is missing, the stage falls back to mono depth. Off by default.
- **No cropping.** The whole reconstruction is kept; the optional room-extent crops (`dense.filter.roi`, `mesh_post.roi`) are off.
- **Plane regularization.** Floor, ceiling and walls are found from face normals and vertices are snapped onto them (`mesh_post.planes.*`).
- **Seam leveling** needs an OpenMVS build with the seam-leveling fix (`mesh_post.texture.seam_leveling`); the `develop` build has it, release 2.4.0 turns textures black.
- **CPU path.** Every CUDA path has a CPU fallback (`--hardware cpu` or `--set device.preference=cpu`).

## Outputs

`outputs/<scene>/runs/<run_id>/07_export/`:

- `model.glb`: textured, Y up; `model_preview.glb`: the same at 300k faces for a quick look. View it at https://gltf-viewer.donmccurdy.com (opened locally) or in Blender with Material Preview shading.
- `obj/model.obj` (Z up), `model_vertex_color.ply`, `model.stl`
- `point_clouds/{sparse,dense}.ply`, `cameras/` (COLMAP model, `transforms.json`), `metadata.json`
- `splats/splats.ply` (`dense.backend=2dgs` only): the trained Gaussians in the standard 3DGS PLY layout; opens in any 3DGS viewer, e.g. drag it into SuperSplat (https://superspl.at/editor) to share it.

Splat viewer (`dense.backend=2dgs`): a real-time browser viewer rendered on the local GPU (gsplat + viser / nerfview):

```bash
pixi run -e gsplat view-splats --run outputs/<scene>/runs/<run_id> [--port 8080] [--mesh]  # then open http://localhost:8080
pixi run -e gsplat view-splats --run outputs/<scene>/runs/<run_id> --check                 # render each mode offscreen, exit
```

The "Splats" panel switches RGB / depth / normal, sets the resolution scale, depth colour range and background, toggles the input camera frustums (click one, or pick a camera number, to jump to it), plays the novel-view path and, with `--mesh`, toggles `mesh_final_geometry` (decimated to 300k faces) over the splats.

Backend comparison of two runs with the same aligned model (time, hole share, geometry, mesh and textured renders with PSNR at held-out cameras, novel views, `recon compare`); what an older run lacks shows as n/a:

```bash
pixi run python scripts/compare_dense.py --a RUN_OPENMVS --b RUN_2DGS --out <dir> [--views 24] [--skip-holes]  # <dir>/index.html
```

`08_report/figures/` holds `overview.png`, `stages.png`, `path.png` (camera path from above with numbered waypoints) and `details.png`, rendered offscreen from the run (config `report.figures`; no OpenGL window: a warning, the run still succeeds). `08_report/index.html` links them and shows the thresholds (registered ratio, reprojection error, gravity residual, untextured faces, wall RMS, dense-to-mesh p95) with pass/fail.

## Run summary and results

```bash
pixi run recon run --input data/hotel1                                     # OpenMVS
pixi run -e gsplat recon run --input data/hotel1 --set dense.backend=2dgs  # 2DGS
```

hotel1 on a GTX 1660 Ti (6 GB), `cuda_6gb` preset. Both runs share everything up to `align` (the 2DGS run `20261118_2dgs` started from a copy of the OpenMVS run `20261105_093901` with `--from dense`), so the cameras are identical and only the dense step and what follows differ. The OpenMVS run is on the code of 2026-11-05. Stage times come from `08_report/summary.json`.

### Shared steps: ingest to align

Both backends start from the same capture, keyframes, SfM model and alignment.

| Stage | Time | Result |
|---|---|---|
| `ingest` | 19 s | 3146 frames, 2 dropped |
| `keyframes` | 79 s | 322 keyframes + 53 stills |
| `sfm` | 927 s | 372 of 375 images registered, 31,566 points, 0.67 px mean reprojection error |
| `align` | 7 s | gravity residual 0.96 deg (median), floor tilt 0.46 deg |

#### Inputs and SfM model

What goes in, and the sparse model SfM builds from it:

![Input keyframes and stills, SfM sparse points and cameras](results/hotel1/inputs.png)

Left: some of the input video keyframes and reference stills. Right: the SfM sparse points with every registered camera in an oblique view; blue dots are video keyframes (camera 1), orange dots the stills (camera 2), arrows the viewing directions.

#### Camera path

How the walk through the room went, seen from above:

![SfM camera path](results/hotel1/sfm_path.png)

The camera path of the video (camera 1, 320 cameras), the sparse points in grey. Numbered circles mark equal time steps along the walk (the table on the right gives the elapsed seconds), black arrowheads the walking direction, thin red arrows the viewing direction of every few cameras, orange squares the stills. The walk starts at the left end of the room (waypoint 1), turns on the spot for about 19 s (waypoints 1 to 5), goes along the lower wall to the middle (6), circles through the middle of the room (7 to 9), and ends with a loop on the right (10 to 12). The path has no jumps: the global mapper cut it into pieces on this capture, so the stage re-mapped it with the incremental mapper (`sfm.mapper.retry_on_jumps`).

### Dense step and after: OpenMVS vs 2DGS

From here on the two backends differ.

| Stage | OpenMVS | 2DGS |
|---|---|---|
| `dense` | 5458 s (91 min): 11.6 M dense points, mesh 4.1 M faces, 2.4 M after refine | about 5040 s (84 min): 500 k Gaussians trained for 24k iterations at 960 px (81 min, 1.5 GB VRAM), TSDF mesh 352 k faces |
| `mesh_post` | 699 s: 2.0 M faces in 136 pieces, wall RMS 0.03 % of the diagonal | 66 s: 352 k faces in 1 piece, wall RMS 0.20 % of the diagonal |
| `export` | 100 s: `model.glb` 156 MB | 17 s: `model.glb` 57 MB, `splats.ply` |
| total | 7289 s (2.0 h) | about 6150 s (1.7 h) |
| surface missing, raw mesh (`hole_share_raw`) | 61 % | 0.8 % |
| surface missing, final mesh (`hole_share_final`) | 6.4 % | 0.7 % |
| held-out photos, splat render | – | PSNR 31.6 dB, SSIM 0.95 |

Hole share: the share of the surface the keyframes see (mono depth of 24 views) that the mesh does not cover; the report computes it for every run. Measured against the OpenMVS dense points instead, which favours OpenMVS, the final meshes miss 3.6 % and 1.8 %. The 2DGS dense step ran as 4 resumed chunks (`dense.gs.max_minutes`); its time is their sum. OpenMVS `dense` finished in 91 min on the OpenMVS `develop` build (the 2.1 build needed about 2.4 times as long for densify alone).

#### OpenMVS results

The OpenMVS outputs from above the room:

![OpenMVS: dense point cloud, mesh, textured model](results/hotel1/openmvs.png)

Left to right: the dense point cloud, the mesh after plane snapping and gap filling, the textured model. The mesh keeps fine detail but has holes where nothing was seen well (floor beside the bed, behind furniture, plain walls), even after the gap filling from mono depth.

#### 2DGS results

The 2DGS outputs, from a held-out camera and from above the room:

![2DGS: held-out photo and splat render, mesh, textured model](results/hotel1/2dgs.png)

Left to right: a photo held out of the training next to the splat render from its pose, the mesh after plane snapping, the textured model. The mesh is closed almost everywhere; it is smoother, with less fine detail and a few thin floaters near the walls. The light grey patches on the floor of the textured model are mesh faces that no input photo sees clearly (for example beside and under the bed), so texturing has no colour for them and fills them flat; the surface is there, it is just not coloured.

#### Side by side at held-out cameras (with the real photo)

Accuracy against ground truth: real camera poses from the walk whose photos were kept out of the 2DGS training, so each render can be compared with what the camera actually saw.

![Held-out cameras: photo, OpenMVS mesh, 2DGS mesh, 2DGS splats](results/hotel1/backends_heldout.png)

One row per held-out camera; columns: the photo, the OpenMVS textured mesh and the 2DGS textured mesh from the same pose (white: no surface), the 2DGS splat render. OpenMVS leaves holes on the wall, window and curtain; 2DGS covers them. OpenMVS did use these photos (only 2DGS held them out), so this comparison slightly favours OpenMVS. These poses are at eye level on the walked path, where every method looks its best.

#### Side by side at novel views (no photo exists)

Completeness from new angles: viewpoints no camera ever took, as when flying around the model.

![Novel views: OpenMVS mesh, 2DGS mesh, 2DGS splats](results/hotel1/backends_novel.png)

One row per viewpoint, along the walk lifted by 10 % of the room height and pitched down 10 degrees; columns: OpenMVS textured mesh, 2DGS textured mesh, 2DGS splat render. Surfaces that were only seen at grazing angles or partly hidden now face the viewer, so the OpenMVS holes are larger here than at the held-out cameras, while 2DGS stays closed.

#### Unseen area in the real-time viewer

What the splats show where no camera looked:

![2DGS splats in the viewer, floor behind the bed](results/hotel1/viewer_unseen.jpg)

The 2DGS splats in the real-time viewer (`view-splats`) from a viewpoint the walk never took. The floor behind and beside the bed, which no camera saw clearly, is filled in with carpet continuing from the visible floor, where the OpenMVS mesh has a hole. It is a plausible fill, not a measurement, and blurrier than the observed surfaces.

All figures of this section and the top image are made with `pixi run -e gsplat python scripts/make_readme_figures.py --backends RUN_OPENMVS RUN_2DGS` (from the two runs' outputs and report panels); `scripts/compare_dense.py` writes the full comparison page (timings, hole masks, geometry, textured PSNR).

## Tests

```bash
pixi run test    # one test file per stage, on a short test clip (scripts/make_test_clip.py)
```

## Tutorials

Small standalone scripts: `pixi run 1a_projection` (projection), `1b_epipolar` (essential/fundamental matrix, epipolar lines), `1c_two_view_geometry` (matching, relative pose, triangulation). Their plots are written to `results/tutorial_*.png`; `pixi run python scripts/make_readme_figures.py --tutorials` joins the epipolar and feature-match plots into the figure below.

![Tutorials: epipolar geometry and feature matches](results/tutorials.png)

Top: epipolar geometry of two views (1b). Bottom: feature matches between two images (1c).
