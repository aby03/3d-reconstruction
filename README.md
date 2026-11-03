# Overview

Hands-on experiments with computer vision concepts and 3d reconstruction.

## Project Setup

First-time setup after cloning. Requires Linux x86_64, an NVIDIA GPU with a CUDA 12.9+ driver (developed on a GTX 1660 Ti, sm_75), `git` and `curl`/`wget`.

1. Install pixi
```bash
wget -qO- https://pixi.sh/install.sh | sh
```
Restart the shell, then check `pixi --version`.

2. Clone with submodules (OpenMVS and VCG live in `third_party/`)
```bash
git clone --recurse-submodules https://github.com/aby03/3d-reconstruction.git
cd 3d-reconstruction
# if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

3. Install the main environment
```bash
pixi install
```

4. Build OpenMVS with CUDA (one-off, takes several minutes)
```bash
pixi install -e build
pixi run -e build openmvs
```
Binaries are installed to `.local/openmvs/bin/OpenMVS/` and are added to `PATH` by pixi. Details and troubleshooting: [OpenMVS CUDA build](plans/architecture/openmvs_cuda_build.md).

5. Verify the OpenMVS build
```bash
pixi run DensifyPointCloud --help      # should print "--cuda-device"
pixi run -e build ldd .local/openmvs/bin/OpenMVS/DensifyPointCloud | grep -Ei 'cuda|not found'
```

6. Download data
```bash
pixi run download-data
```

## Running 3d reconstruction pipeline

The pipeline is run through the `recon` pixi task. Use the ~5 s test clip `data/room2_tiny` for quick sanity checks: every step runs in seconds on it, while the full `data/room2` capture gets expensive from SfM onward. Create the clip once with `pixi run make-test-clip`.

```bash
pixi run recon --help
pixi run recon run --help
pixi run recon list-stages     # stages in pipeline order, with their directories and config sections
```

### Run the pipeline
```bash
pixi run recon run --input data/room2_tiny                     # every stage: ingest ... dense, mesh_post, export
pixi run recon run --input data/room2_tiny --to keyframes      # stop after a stage
pixi run recon run --input data/room2_tiny --set keyframes.method=uniform
```
Per-sequence stages (ingest, keyframes) write to `outputs/<scene>/sequences/<sequence_id>/<NN_stage>/` (here `outputs/room2_tiny/sequences/room2_tiny_20261021_211400/`); scene stages (sfm, align, dense, mesh_post, export), which combine every sequence of a run, write to the run dir `outputs/<scene>/runs/<run_id>/<NN_stage>/`. A stage whose result is up to date (finished successfully with the same config section and the same upstream results, and every artifact still there) is skipped, so a second run takes about a second and changing a `keyframes.*` key re-runs only keyframes. Each invocation leaves a record in `outputs/<scene>/runs/<run_id>/`: `run_manifest.json` (status of every stage: `success`, `skipped`, `failed`), `config.resolved.yaml` (the merged config) and `logs/pipeline.log`. Every `recon run` starts a new run dir, so its scene stages run again; `--resume` continues in an existing run dir instead and skips its up-to-date scene stages:
```bash
pixi run recon run --input data/room2_tiny --resume                    # the latest run
pixi run recon run --input data/room2_tiny --resume 20261031_025715    # a given run id
```

| Option | Meaning |
|---|---|
| `--input DIR` | capture folder (or its stamp folder) |
| `--scene NAME` | scene name, defaults to the capture folder name |
| `--output-root DIR` | same as `--set run.output_root=DIR`, defaults to `outputs` |
| `--from STAGE` / `--to STAGE` | first / last stage to run |
| `--only STAGE` | run just this stage; its upstream stages must already be done |
| `--force STAGE` | re-run this stage and every stage after it |
| `--resume [RUN_ID]` | continue in that run dir (no id: the latest run) instead of a new one; it must be a run of the same capture |
| `--hardware PRESET` | `configs/hardware/<PRESET>.yaml` between the defaults and `--config`; `auto` (default) picks one by GPU memory, `none` uses none |
| `--config FILE` | YAML file layered on `configs/default.yaml` and the hardware preset (repeatable) |
| `--set KEY=VALUE` | override one key, value parsed as YAML (repeatable), e.g. `--set ingest.gravity_method=lowpass` |
| `--gravity-method`, `--time-offset-ns`, `--no-verify-refs` | shortcuts for `ingest.gravity_method`, `ingest.camera_time_offset_ns`, `loader.verify_reference_jpegs=false` |

**Config.** `configs/default.yaml` lists every key with its default: `run`, `device` and one section per step (`loader`, `ingest`, `keyframes`, `sfm`, `align`, `dense`, `mesh_post`, `export`). Unknown keys and wrong types are errors that name the full key path. Layers, later wins: `default.yaml`, the hardware preset, `--config` files, `--set`.

**Hardware presets.** `configs/hardware/cuda_6gb.yaml` (4–10 GB of GPU memory: the defaults written out), `cuda_large.yaml` (10 GB or more: full-resolution depth maps, 8 neighbour views) and `cpu.yaml` (no CUDA OpenMVS, or `device.preference: cpu`: everything on the CPU, depth maps at a quarter of the image size). `--hardware auto` (the default) picks one by the probed GPU, `--hardware cpu` forces a preset, `--hardware none` skips the layer. The preset is named in the first line of the run's `config.resolved.yaml` and under `hardware` in `run_manifest.json`.

**GPU.** `device.preference: auto` (default) runs an operation on CUDA where this machine supports it and falls back to the CPU with a warning; `cuda` makes a missing CUDA path an error, `cpu` forces the CPU (`--set device.preference=cpu`). Per operation: `loader.decode_backend` (`nvdec`/`pyav`) and `keyframes.metrics_backend` (`torch`/`numpy`); SfM (features, matching, mapping) and the OpenMVS tools of the dense and mesh_post steps follow `device.preference`. Each stage manifest records the backends it used under `backends`.

### Check the machine
```bash
pixi run recon doctor        # GPU, which operations can use CUDA, versions of Python packages and tools
```

### Inspect a capture (writes nothing)
Loads the capture, checks frame timing, sensor streams and reference JPEGs, syncs the sensors to the frames and prints a summary.
```bash
pixi run recon inspect --input data/room2_tiny
```
Takes `--config`, `--set` and the capture shortcuts (`--gravity-method`, `--time-offset-ns`, `--no-verify-refs`).

### Step 1: ingest
```bash
pixi run recon ingest --input data/room2_tiny
```
Same as `recon run --only ingest`, with the same options except `--from/--to/--only`; `--force` re-runs it even if it is up to date. Writes `outputs/room2_tiny/sequences/room2_tiny_20261021_211400/01_ingest/`.

Results to look at:
```bash
D=outputs/room2_tiny/sequences/room2_tiny_20261021_211400/01_ingest
cat $D/stage_manifest.json $D/reports/sync_report.json
head $D/frames.csv $D/sync/frame_sensor_data.csv
ls $D/reports/plots/        # timeline, sensor_rates, gravity, omega
```

### Step 2: keyframes
Needs a finished ingest of the same capture (`recon ingest` first, or use `recon run`).
```bash
pixi run recon keyframes --input data/room2_tiny
pixi run recon keyframes --input data/room2_tiny --set keyframes.method=uniform --uniform-fps 2   # uniform baseline
```
Same as `recon run --only keyframes`. Scores every frame for blur, exposure, texture and rotation speed (`frame_scores.csv`), rejects bad frames and picks keyframes: by default (`keyframes.method=adaptive`) a new keyframe is taken when the image has moved enough since the last one (KLT flow, lost tracks) or the gyro has turned 8°, with the thresholds tuned so there are 2.5–7 keyframes per second and never a gap above 1 s; `--set keyframes.method=uniform` instead picks the best frame around each 1/`uniform_fps` tick. It writes the keyframes as full-resolution upright JPEGs, and writes per-keyframe IMU priors (gravity in the camera frame, gyro rotations between nearby keyframes) to `.../room2_tiny_20261021_211400/02_keyframes/`. `--force` re-runs it even if it is up to date. Reject thresholds and score weights are `keyframes.reject.*` and `keyframes.weights.*`.

Results to look at:
```bash
D=outputs/room2_tiny/sequences/room2_tiny_20261021_211400/02_keyframes
cat $D/stage_manifest.json $D/reports/selection_report.json
head $D/keyframes.csv $D/frame_scores.csv $D/priors/gravity_cam.csv $D/priors/relative_rotations.csv
ls $D/images/cam0/          # <frame idx>.jpg
ls $D/reports/plots/        # scores_timeline.png (scores, flow, tracks; keyframes in red), keyframe_strip.jpg
```

### Step 3: sfm
Needs finished keyframes (`recon keyframes` first, or use `recon run`).
```bash
pixi run recon sfm --input data/room2_tiny
pixi run recon sfm --input data/room2_tiny --set sfm.matching.strategy=sequential --force
```
Same as `recon run --only sfm --resume`: it continues in the latest run of the scene (or starts a new run when there is none), so a finished SfM is kept; `--resume RUN_ID` picks another run and `--force` re-runs it. It extracts SIFT features of every keyframe (GPU), matches them (`sfm.matching.strategy`: `auto` matches every pair up to 200 keyframes and each keyframe with its 20 successors beyond that) and runs COLMAP's global mapper. The IMU helps: each keyframe's gravity direction (`priors/gravity_cam.csv`) is written into the COLMAP database as a pose prior and constrains rotation averaging (`sfm.priors.use_gravity_prior`; the model's +y then points roughly down), and before mapping every verified image pair whose relative rotation disagrees with the gyro by more than 10° (pairs over 5 s apart: whose rotation doesn't carry one image's gravity onto the other's within 5°) is removed (`sfm.priors.imu_pair_filter.*`; if more than 30% disagree, nothing is removed and a warning is logged). With sequential matching, every 10th keyframe is also matched with its 50 most similar keyframes (COLMAP's vocabulary tree, downloaded once into `~/.cache/colmap/`), so revisits of a room are connected (`sfm.matching.loop_detection*`, `vocab_tree_path`). If the global mapper registers fewer than 80% of the keyframes or fails, it is retried without gravity and then with COLMAP's incremental mapper, and the best result is kept (`sfm.mapper.fallback`, `min_registered_ratio`; the report lists every attempt). The mapped model (`sparse/raw`, a `SIMPLE_RADIAL` camera per sequence) is then refined into `sparse/refined`, **the model later steps use**: keyframes observing fewer than 15 points are deregistered (their pose is a guess), points are retriangulated, the camera becomes `OPENCV`, then bundle adjustment, removal of observations above 2 px, short tracks and small triangulation angles, and a final bundle adjustment (`sfm.refine.*`; `enabled: false` copies the raw model). Implausible intrinsics or trajectory jumps are reported as warnings. Everything is written to `outputs/room2_tiny/runs/<run_id>/03_sfm/`. COLMAP's own log goes to `runs/<run_id>/logs/sfm/colmap.log.<timestamp>`. Keys: `sfm.initial_camera_model`, `sfm.features.*` (`max_num_features`, `max_image_size`, `domain_size_pooling`), `sfm.matching.*` (`strategy`, `exhaustive_max_images`, `overlap`, `quadratic_overlap`, `min_num_inliers`, `max_num_matches`, `loop_detection`, `loop_detection_period`, `loop_detection_num_images`, `vocab_tree_path`), `sfm.priors.*` (`use_gravity_prior`, `gravity_min_conf`, `imu_pair_filter.{enabled, max_rot_err_deg, max_dt_s, tilt_err_deg, max_removed_frac}`), `sfm.mapper.{backend, fallback, min_registered_ratio}`, `sfm.refine.{enabled, final_camera_model, refine_principal_point, max_reproj_px, min_track_len, min_tri_angle_deg, min_image_obs}`. Without the IMU:
```bash
pixi run recon sfm --input data/room2_tiny --force --set sfm.priors.use_gravity_prior=false --set sfm.priors.imu_pair_filter.enabled=false
```

Results to look at:
```bash
D=$(ls -d outputs/room2_tiny/runs/*/03_sfm | tail -1)
cat $D/stage_manifest.json $D/reports/sfm_report.json $D/intrinsics.json
ls $D/sparse $D/reports/plots/       # raw, refined, refined_txt; track_len_hist, reproj_hist, trajectory_topdown
pixi run colmap gui --database_path $D/database.db --image_path $D/images --import_path $D/sparse/refined
```
`reports/pair_filter.csv` lists every verified pair with its check (`gyro`, `tilt`, `none`), angle error and whether it was removed; `gravity_spread_deg` in the report is how far the registered cameras' gravity directions spread around their mean in the model (small = model and IMU agree). `intrinsics.json` and `sparse_points.ply` (coloured sparse points) come from the refined model; `images/` holds one symlink per sequence to its keyframe images (the COLMAP image root).

### Step 4: align
Needs a finished SfM (`recon sfm` first, or use `recon run`).
```bash
pixi run recon align --input data/room2_tiny
pixi run recon align --input data/room2_tiny --force --set align.gravity.method=camera_up   # without the IMU
```
Same as `recon run --only align --resume`: like `recon sfm` it continues in the latest run (`--resume RUN_ID` picks another, `--force` re-runs it). It moves the refined SfM model (`03_sfm/sparse/refined`) into the world frame W: **Z up, gravity along −Z, the floor at z = 0, the origin under the median camera position, X/Y along the dominant walls**. Units stay those of the SfM model (`units: "arbitrary"`; no metric scale yet). Gravity is always estimated, never read off the model's axes: each registered keyframe's IMU gravity (`priors/gravity_cam.csv`, confidence ≥ `align.gravity.min_conf`) is rotated into the model, and their confidence-weighted mean, made robust with a Huber reweighting (`huber_deg`), is rotated onto −Z (`align.gravity.method: imu`). Without enough priors (`auto`: fewer than 3, or fewer than `min_prior_frac` of the registered keyframes) or with `method: camera_up`, the mean image-down direction of the cameras is used instead and replaced by the floor normal (a warning says so). The floor is the lowest big peak of the point-height histogram below the cameras, fitted with RANSAC and least squares (`align.floor.*`; distances are fractions of the scene size since the units are arbitrary; RANSAC is seeded with `run.seed`); a floor normal within `max_tilt_correction_deg` (2°) of gravity is rotated onto Z. No floor → z = 0 at the 2nd percentile of the point heights, with a warning. Last, the wall direction: normals of the sparse points (PCA of their 30 nearest neighbours) that are planar and near-horizontal vote in a histogram of their direction mod 90°; the peak becomes a rotation about Z (X along the longer wall direction unless the two are within 10%), applied only when at least `min_confidence` (25%) of the votes fall in the peak's 1° bin or the 2 bins either side (`align.manhattan.*`; the step is listed with `applied: false` otherwise). The camera–IMU calibration is also checked per sequence (`align.cam_imu.*`): from keyframe pairs up to 5 apart, the SfM rotations against the gyro give a hand-eye estimate of `R_cam_imu` (angle to the nominal one, conditioning) and the camera time offset (±100 ms, 1 ms steps, the gyro track slerped at the shifted keyframe times). These are **report only** (`suggested_config`, only for a well-conditioned estimate with a clear minimum): to use a suggested time offset, re-run from ingest with `--time-offset-ns` (as a new scene); `R_cam_imu` can't be set from the config yet. Writes `outputs/room2_tiny/runs/<run_id>/04_alignment/`: the aligned model `sparse_aligned/` (**the model later steps use**) and `sparse_aligned_txt/`, `sparse_points_aligned.ply` (coloured), `transform.json` (`W_from_sfm` as scale/R/t with `x_W = scale · R · x_sfm + t`, and the `steps` it is composed of, applied in list order), `trajectory_tum/<sequence_id>.txt` (TUM `t tx ty tz qx qy qz qw`, camera-to-world, t in seconds on the capture clock of that sequence) and the report. Keys: `align.gravity.{method, huber_deg, min_conf, min_prior_frac, max_residual_deg}`, `align.floor.{enabled, min_inlier_frac, max_normal_deg, cameras_above_frac, max_tilt_correction_deg, fallback_max_deg, bin_frac, ransac_iterations, max_reproj_px, min_track_len}`, `align.manhattan.{enabled, min_confidence, knn, min_planarity, max_normal_z, peak_halfwidth_deg, extent_tie}`, `align.cam_imu.{enabled, pair_span, min_rotation_deg, max_offset_ms}`.

Results to look at:
```bash
D=$(ls -d outputs/room2_tiny/runs/*/04_alignment | tail -1)
cat $D/stage_manifest.json $D/reports/alignment_report.json $D/transform.json
head $D/trajectory_tum/*.txt
ls $D/reports/plots/        # gravity_residuals, height_hist (floor line, cameras), trajectory_topdown (W x/y), yaw_hist
```
In the report, `gravity.residual_median_deg` is how far the keyframes' IMU gravity is from −Z after alignment (about 1°: the image-vs-IMU noise), `floor.tilt_to_gravity_deg` the angle between the floor normal and the IMU gravity before it was corrected, and `gravity.down_vs_model_y_deg` how far the SfM model's +y was from gravity. `manhattan` has the wall direction, its confidence and whether the yaw was applied; `cam_imu.<sequence_id>` has `angle_to_nominal_deg` (hand-eye `R_cam_imu` vs the nominal one), `well_conditioned`, `time_offset_ms` with `cost_at_zero`/`cost_best` and `corr_peak_ms` (an angular-speed cross-check), and `suggested_config` the values worth trying. Skip the yaw with `--set align.manhattan.enabled=false`. Open `sparse_aligned_txt/` or the PLY in a viewer (e.g. MeshLab): the floor is flat at z = 0 with Z up.

### Step 5: dense
Needs a finished alignment (`recon align` first, or use `recon run`) and the OpenMVS build (setup step 4; `pixi run recon doctor` shows whether it has CUDA).
```bash
pixi run recon dense --input data/room2_tiny
pixi run recon dense --input data/room2_tiny --force --set device.preference=cpu   # the CPU path
pixi run recon dense --input data/room2_tiny --hardware cpu                         # the CPU preset (quarter-size depth maps)
```
Same as `recon run --only dense --resume`: like `recon align` it continues in the latest run (`--resume RUN_ID` picks another, `--force` re-runs it). It densifies the aligned model (`04_alignment/sparse_aligned`), so every output is in the world frame W (Z up, floor at z = 0, units of the SfM model). First the model and its keyframes are undistorted with pycolmap into PINHOLE cameras (`undistorted/`, COLMAP layout, image names unchanged, black border cropped: 1080×1920 → 1043×1857). Then the OpenMVS tools run in `05_dense/openmvs/`: `InterfaceCOLMAP` (→ `scene.mvs`), `DensifyPointCloud` (patch-match depth maps, fused into `scene_dense.mvs` + `scene_dense.ply`), `ReconstructMesh` (→ `scene_dense_mesh.ply`), `RefineMesh` (photometric refinement, → `scene_dense_mesh_refine.ply`) and `TextureMesh` (a quick low-resolution textured preview into `05_dense/preview/`, seam leveling off). With `device.preference: auto` they run on the GPU when the OpenMVS build has CUDA (`--cuda-device <device.gpu_index>`); the CPU path passes `--cuda-device -2` to every tool (on tiny: densify 70 s on CUDA, 126 s on the CPU). RefineMesh runs on the CPU by default (`dense.refine.use_cuda: false`: on room2 the CPU took 16.7 min, CUDA 20.3 min and 3.2 GB of GPU memory, same mesh); `true` makes it follow `device.preference`. If a CUDA tool runs out of GPU memory, densify is retried once a resolution level lower with 2 neighbour views fewer, and with `device.preference: auto` a tool that still fails that way runs again on the CPU (warnings `oom_retry: ...` / `oom_cpu_fallback: ...`, listed under `oom_retries` in the report). RefineMesh is stopped after `dense.refine.max_minutes`; `mesh_refined.ply` is then a copy of the raw mesh and a warning says so (`dense.refine.enabled: false` copies it without a warning; `dense.preview.enabled: false` skips the preview). The `.mvs` files store the image paths relative to `openmvs/` (`../undistorted/images/...`), so don't move `openmvs/` or `undistorted/`. OpenMVS's region-of-interest crop is off (`dense.openmvs.crop_to_roi: false`): its estimated ROI cuts the upper walls. Images are processed at half size (`resolution_level: 1`), but OpenMVS never goes below `min_resolution` on the long side, so on 1857 px keyframes level 2 is the same as level 1 unless `min_resolution` is lowered too; the report gives the effective long side. Depth maps are removed after fusion (`keep_dmaps: false`). Each tool's output goes to `outputs/<scene>/runs/<run_id>/logs/dense/<Tool>.log` (and OpenMVS's own `<Tool>-<stamp>.log` files are moved there too; pycolmap's to `colmap.log.*`); a failing tool stops the stage with the end of its log in the error. Keys: `dense.undistort.max_image_size`, `dense.openmvs.{resolution_level, min_resolution, max_resolution, number_views, number_views_fuse, sub_resolution_levels, iters, geometric_iters, filter_point_cloud, crop_to_roi, keep_dmaps, bin_dir}` (`bin_dir: null` takes OpenMVS from `PATH`), `dense.mesh.{remove_spurious, remove_spikes, close_holes, smooth, decimate}`, `dense.refine.{enabled, use_cuda, resolution_level, max_face_area, scales, regularity_weight, max_minutes}`, `dense.preview.{enabled, resolution_level, min_resolution}`.

Results to look at:
```bash
D=$(ls -d outputs/room2_tiny/runs/*/05_dense | tail -1)
cat $D/stage_manifest.json $D/reports/dense_report.json
ls $D/reports/plots/        # dense_height_hist (floor spike at z = 0, walls up to the ceiling), dense_topdown (W x/y)
ls $D/../logs/dense/
```
Writes `dense_points.ply` (the fused cloud as a plain binary PLY: `x y z nx ny nz red green blue`), `mesh_raw.ply` (the ReconstructMesh surface), `mesh_refined.ply` (the refined surface, decimated by RefineMesh to about a third of the faces; the mesh later steps start from), `preview/mesh.obj` (+ `mesh.mtl` and one texture atlas jpg; the unused atlas space is orange), `undistorted/` (images, `sparse/` PINHOLE model, `stereo/`; the texturing input of later steps), `openmvs/` (`scene.mvs`, `scene_dense.mvs` with the `scene_dense.ply` it references, `scene_dense_mesh.ply`) and the report. In the report, `num_dense_points`, `dense.inside_sparse_bbox_frac` and `mesh_raw.inside_sparse_bbox_frac` (share of the vertices inside the box of the aligned sparse points and cameras; ≥ 0.99 expected, a warning below), `mesh_raw.largest_component_frac` (share of the faces in the largest connected piece; > 0.9 expected), the same for the refined mesh under `refine` (with `ran`, `timed_out`), `preview` (faces, atlas size), `dense.floor_band_frac` / `floor_band_median_z` (points within 1% of the scene diagonal of z = 0), `densify.effective_long_side`, and per tool its time, peak GPU memory (`peak_vram_mb`, sampled with `nvidia-smi`, GPU-wide) and log. Open `dense_points.ply`, `mesh_raw.ply`, `mesh_refined.ply` and `preview/mesh.obj` in MeshLab (Z up; `openmvs/scene_dense.ply` also loads there, but trimesh can't read its per-point view lists).

### Step 6: mesh_post
Needs a finished dense step (`recon dense` first, or use `recon run`).
```bash
pixi run recon mesh_post --input data/room2_tiny
pixi run recon mesh_post --input data/room2_tiny --force --hardware cpu   # texturing on the CPU
```
Same as `recon run --only mesh_post --resume`: it continues in the latest run (`--resume RUN_ID` picks another, `--force` re-runs it). It starts from `05_dense/mesh_refined.ply` (`mesh_post.input: raw` takes `mesh_raw.ply`) and, in the world frame W:
1. **Cleanup** with pymeshlab → `mesh_clean.ply`: duplicate vertices and faces, zero-area faces and unreferenced vertices removed, non-manifold edges and vertices repaired, then every connected piece with fewer than `cleanup.min_component_frac` (0.5%) of the faces or smaller than `cleanup.min_component_diam_frac` (2%) of the bounding-box diagonal dropped (tiny: 3 → 1 pieces, 32 faces removed).
2. **Small holes** → `mesh_holes_filled.ply`: the mesh's boundary loops (edges used by one face) are found; a loop longer than `holes.max_loop_perimeter_frac` (30%) of the bounding-box perimeter is the outer edge of a partly captured room and stays open, and a loop with more than `holes.small_hole_max_edges` (100) edges stays open too. The remaining loops are triangulated from their own vertices with pymeshlab (no new vertices). When nothing is closed the file is a copy of `mesh_clean.ply` (tiny and room2 have one loop each, the outer one).
3. **Decimation** → `mesh_final_geometry.ply`: quadric edge collapse of `mesh_holes_filled.ply` down to `decimate.texture_face_budget` (600k) faces, only when the mesh has more (room2: 728k → 600k; tiny's 101k is copied unchanged).
4. **Texturing** with OpenMVS `TextureMesh` → `textured/mesh.obj` + `mesh.mtl` + `mesh_material_NN_map_Kd.jpg` (one JPEG atlas per 8192² of texture; tiny: one 4096² atlas, room2: one 8192²). It runs in `05_dense/openmvs/` on the full-resolution undistorted keyframes (`texture.resolution_level: 0`), on CUDA like the dense tools (CPU with `device.preference: cpu`, and as a fallback after a GPU out-of-memory error under `auto`). Seam leveling is always off (it turns large parts of the atlas black in OpenMVS 2.4.0), and holes are not closed, so the textured mesh has exactly the faces of `mesh_final_geometry.ply`. Atlas space no view covers is magenta (`0xFF00FF`).
5. **Geometry check**: the distance from the dense points (`05_dense/dense_points.ply`, a seeded sample of up to 200k) to the surface of `mesh_final_geometry.ply`, and the share of the mesh area farther than 2τ from every dense point (surface the mesher bridged across space no keyframe saw). τ is 0.5% of the scene diagonal (`diag` in the report) in arbitrary units, 2 cm in metres. A dense-to-mesh p95 above 2τ is a warning.

Keys: `mesh_post.input`, `mesh_post.cleanup.{min_component_frac, min_component_diam_frac}`, `mesh_post.holes.{small_hole_max_edges, max_loop_perimeter_frac}`, `mesh_post.decimate.{texture_face_budget, planar_quadric}`, `mesh_post.texture.{resolution_level, min_resolution, max_texture_size, outlier_threshold, cost_smoothness_ratio, virtual_face_images, sharpness_weight, patch_packing_heuristic}`. Tool logs go to `outputs/<scene>/runs/<run_id>/logs/mesh_post/`.

Results to look at:
```bash
D=$(ls -d outputs/room2_tiny/runs/*/06_mesh_post | tail -1)
cat $D/stage_manifest.json $D/reports/mesh_report.json
ls $D/textured/
```
In `mesh_report.json`: `cleanup.before`/`after` (vertices, faces, connected pieces, boundary and non-manifold edges), `holes` (`loops`, `filled`, `filled_faces`, `skipped_outer`, `skipped_large`, `largest_loop_edges`), `decimate` (faces in/out, budget), `final` (the texturing input: pieces, share inside the sparse bounding box, ≥ 0.99 expected), `texture` (`atlas_count`, `atlas_size`, `atlas_empty_frac`: magenta share of the atlas, `black_texel_frac`: near-black share of the used texels, `untextured_face_frac`: faces whose texture is magenta, a warning above 1%; tiny 0.9%, room2 1.0%), `qa` (`tau`, `dense_to_mesh` mean/RMS/p50/p95/max in scene units, `p95_frac`: p95 as a share of the diagonal, also the manifest metric `qa_p95_frac`, tiny 0.22%; `unsupported_area_frac`, tiny 23%), time and GPU memory per tool. Open `textured/mesh.obj` in MeshLab (Z up): walls and floor carry the photos, magenta marks faces no keyframe saw. OpenMVS's MTL says `Tr 1.000000`, which some viewers read as fully transparent; the exported `07_export/obj/` copy fixes that.

### Step 7: export
Needs a finished mesh_post step.
```bash
pixi run recon export --input data/room2_tiny
```
Same as `recon run --only export --resume`. Turns `06_mesh_post/textured/` into the deliverables in `07_export/`:
- `model.glb`: binary glTF, **Y up** (the vertices are rotated `x_G = (x_W, z_W, −y_W)`, no node transform), the JPEG atlases embedded unchanged (no external files), one PBR material per atlas (base-colour texture, metallic 0, roughness 1, opaque, double-sided so the room shows from inside; `export.glb.double_sided`).
- `obj/model.obj`, `obj/model.mtl`, `obj/textures/model_NN.jpg`: the same mesh **Z up** (world W), MTL made opaque (`Tr 0`, `d 1`); the first line of the OBJ says `# world W: +Z up, units <units>`.
- `metadata.json`: scene, run id, sequences, `pipeline_version` (git commit), `units` (`arbitrary` unless the alignment set a metric scale: a viewer's "metres" are then only a convention), the world frame, per file its up axis, faces, vertices, texture sizes, bytes and sha256, the W bounding box and the number of cameras.

Every file is reloaded and checked (GLB header and chunk lengths, no external URIs, face count and bounding box against the source); a failed check fails the stage. Tiny: 8.5 MB GLB in 1 s; room2: 50 MB.

**Viewing the GLB.** Drag `07_export/model.glb` into https://gltf-viewer.donmccurdy.com (the file is opened locally in the browser, nothing is uploaded) or import it in Blender (File → Import → glTF 2.0). Blender opens in Solid shading, which draws the mesh grey: switch the viewport to Material Preview (Z key, or the shading buttons in the viewport header) to see the texture. The floor is at y = 0 and the room stands upright; orbit inside it to see the walls.
```bash
ls -d outputs/room2_tiny/runs/*/07_export/model.glb | tail -1
cat $(ls -d outputs/room2_tiny/runs/*/07_export | tail -1)/metadata.json
```

### Tests
```bash
pixi run test               # runs every step's test on data/room2_tiny
```


## Running tutorials

### 1a. projection
Project 3d world coordinates to image plane given intrinsics and extrinsics
```
pixi run 1a_projection
```

![3D point projection](results/tutorial_1a_projection.png)

### 1b. Epipolar Geometry
Covers Essential Matrix, Fundamental Matrix and Epipolar line computation given 2 views sharing world coordinates frame.
```
pixi run 1b_epipolar
```

![Epipolar geometry](results/tutorial_1b_epipolar.png)

### 1c. Multi-view Geometry
Using 2 images, compute matching points, essential matrix, relative pose, camera positions and matching points projection in 3D.
```
pixi run 1c_two_view_geometry
```

![Two-view feature matches](results/tutorial_1c_matches.png)

![Triangulated 3D point cloud](results/tutorial_1c_point_cloud.png)

### 3. Room Reconstruction
Compare two room reconstruction pipelines. The `3a` baseline uses COLMAP for sparse-to-dense reconstruction. The `3b` pipeline samples the `room2` MP4 with OpenCV, estimates camera poses with COLMAP, and uses OpenMVS for dense reconstruction and image texturing.

```bash
pixi run 3a_reconstruction
pixi run 3b_reconstruction
```

The `3b` pipeline defaults to `data/room2` and writes timestamped runs under `outputs/room2_reconstruction/`. OpenMVS must be built first (setup step 4); the pipeline finds its tools on `PATH`. See the [reconstruction guide](src/tutorials/3_reconstruction/README.md) for details.