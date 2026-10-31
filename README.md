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
pixi run recon ingest --help
```

### Inspect a capture (writes nothing)
Loads the capture, checks frame timing, sensor streams and reference JPEGs, syncs the sensors to the frames and prints a summary.
```bash
pixi run recon inspect --input data/room2_tiny
```

### Step 1: ingest
```bash
pixi run recon ingest --input data/room2_tiny
```
Writes `outputs/<scene>/sequences/<sequence_id>/01_ingest/` (here `outputs/room2_tiny/sequences/room2_tiny_20261021_211400/01_ingest/`) and logs to `.../logs/pipeline.log`. A second run keeps the finished result; add `--force` to re-run.

| Option | Meaning |
|---|---|
| `--input DIR` | capture folder (or its stamp folder) |
| `--scene NAME` | scene name, defaults to the capture folder name |
| `--output-root DIR` | defaults to `outputs` |
| `--force` | re-run even if a successful result exists |
| `--gravity-method complementary\|lowpass` | gravity estimator, default `complementary` |
| `--time-offset-ns N` | added to frame times before sensor lookup |
| `--no-verify-refs` | skip the reference JPEG check (faster, no pixel decoding) |

`inspect` takes the same capture options (`--input`, `--gravity-method`, `--time-offset-ns`, `--no-verify-refs`).

Results to look at:
```bash
D=outputs/room2_tiny/sequences/room2_tiny_20261021_211400/01_ingest
cat $D/stage_manifest.json $D/reports/sync_report.json
head $D/frames.csv $D/sync/frame_sensor_data.csv
ls $D/reports/plots/        # timeline, sensor_rates, gravity, omega
```

### Step 2: keyframes
Needs a finished ingest of the same capture (`recon ingest` first).
```bash
pixi run recon keyframes --input data/room2_tiny
```
Selects keyframes uniformly in time, writes them as full-resolution upright JPEGs, and writes per-keyframe IMU priors (gravity in the camera frame, gyro rotations between nearby keyframes) to `.../room2_tiny_20261021_211400/02_keyframes/`. A second run keeps the finished result; add `--force` to re-run.

| Option | Meaning |
|---|---|
| `--input DIR` | capture folder (or its stamp folder) |
| `--scene NAME` | scene name, defaults to the capture folder name |
| `--output-root DIR` | defaults to `outputs` |
| `--uniform-fps F` | keyframes per second of video, default 3 |
| `--force` | re-run even if a successful result exists |

Results to look at:
```bash
D=outputs/room2_tiny/sequences/room2_tiny_20261021_211400/02_keyframes
cat $D/stage_manifest.json $D/reports/selection_report.json
head $D/keyframes.csv $D/priors/gravity_cam.csv $D/priors/relative_rotations.csv
ls $D/images/cam0/          # <frame idx>.jpg
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