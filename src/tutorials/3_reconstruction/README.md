# Room Reconstruction

Run a COLMAP sparse-to-dense reconstruction from the Open Camera Sensors capture. The input defaults to `data/room`; the runner finds its single directory of timestamped images automatically.

```bash
pixi run 3_reconstruction
```

The default pipeline uses timestamp-ordered frames, exhaustive pair matching, SIFT features, an incremental mapper, high-quality dense depth, Delaunay meshing, and one shared `SIMPLE_RADIAL` camera model. Since the capture currently has no camera-intrinsics sidecar or EXIF calibration, COLMAP estimates the shared intrinsics. The accelerometer, gyroscope, and magnetometer CSVs are recorded in the run manifest but are not fused into camera poses yet; correct sensor timestamp alignment, axis convention, and camera-to-IMU calibration are prerequisites for that.

Each run gets a new directory under `outputs/room_reconstruction/` containing the COLMAP database, sparse model, dense workspace/depth maps, fused PLY, Delaunay mesh PLY, GLB point cloud, GLB mesh, log, manifest, and summary. Use `fused_point_cloud.glb` to inspect dense samples or `dense_mesh.glb` to inspect the triangulated surface.

Options:

```bash
pixi run 3_reconstruction -- --quality high
pixi run 3_reconstruction -- --quality extreme
pixi run 3_reconstruction -- --image-dir data/room/20261021_171938 --output-dir outputs/room_reconstruction/test-run
pixi run 3_reconstruction -- --data-type video
pixi run 3_reconstruction -- --camera-model PINHOLE --camera-params fx,fy,cx,cy
pixi run 3_reconstruction -- --cpu
```

The Delaunay mesh is a surface estimate, not guaranteed to be watertight. Check the run summary and COLMAP log for registered-image count and reconstruction quality; self-calibration can fail on blurry, low-texture, or low-parallax captures. Increase to `--quality extreme` for a denser but more expensive run.