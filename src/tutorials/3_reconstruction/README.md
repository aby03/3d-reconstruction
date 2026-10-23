# Room Reconstruction

## 3a: COLMAP baseline

The original COLMAP sparse-to-dense pipeline is preserved in `3a_run.py` and writes runs under `outputs/room_reconstruction/`.

```bash
pixi run 3a_reconstruction
pixi run 3a_reconstruction -- --quality extreme
pixi run 3a_reconstruction -- --image-dir data/room/20261021_171938
```

COLMAP estimates shared camera intrinsics when no calibration is provided. IMU CSV files are recorded but not fused into poses.

## 3b: OpenCV, COLMAP, and OpenMVS

The new pipeline defaults to the MP4 in `data/room2`. OpenCV samples it at 2 frames per second, measures each sampled frame's sharpness using variance of the Laplacian, and stages selected images into a clean image-only directory. All sampled frames are kept by default. COLMAP uses sequential matching for the ordered video frames and estimates camera poses. Before import, COLMAP undistorts registered images and exports PINHOLE cameras, which OpenMVS requires. OpenMVS then densifies the scene, reconstructs and refines a mesh, and projects image textures onto it. The provided extracted JPEGs remain available as an image-only alternative.

OpenCV is provided by Pixi. OpenMVS is not available in the configured conda-forge channel; install its CLI tools from the [official OpenMVS releases](https://github.com/cdcseacave/openMVS/releases/latest) and make `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, and `TextureMesh` available on `PATH`.

```bash
pixi run 3b_reconstruction
pixi run 3b_reconstruction -- --resolution-level 0
pixi run 3b_reconstruction -- --min-sharpness 30
pixi run 3b_reconstruction -- --image-dir data/room2/20261021_211400
pixi run 3b_reconstruction -- --images-only
pixi run 3b_reconstruction -- --sample-fps 1
```

OpenMVS resolution level `0` uses full resolution, `1` half resolution, and `2` quarter resolution; full resolution can require substantially more memory. The final GLB embeds its texture and is packaged as a self-contained file. The OBJ uses its adjacent MTL and JPG texture sidecar. Runs go under `outputs/room2_reconstruction/` with the following layout:

```text
<run>/
	images/                 sampled input frames
	sparse/                 COLMAP sparse camera model
	openmvs/
		scene_textured.obj    textured mesh
		scene_textured.mtl    OBJ material
		scene_textured_material_00_map_Kd.jpg
		scene_textured.glb   self-contained textured mesh
		intermediate/         OpenMVS scenes, depth maps, meshes, logs, undistorted images
	reconstruction.log
	run_manifest.json
	summary.json
```

IMU streams are preserved but not fused; doing so requires verified time alignment, sensor axes, and camera-to-IMU calibration.