# Project notes

This is a hobby project for trying out and playing with different 3D reconstruction methods. The goal is to capture with devices people already have, like a phone, and to reconstruct on ordinary compute, like a decent laptop with a mid-range GPU.

Off-the-shelf tools (COLMAP/GLOMAP for camera poses, OpenMVS for dense reconstruction and texturing) do the heavy lifting. The fun part is making them work on real handheld indoor captures, which have blank walls, turns in place and a phone's IMU. The IMU turned out to be useful for sanity checks rather than as a direct pose source.

## What I've played with so far

- **Matching:** pairing frames by capture time instead of file name. It's cheap and registers more images.
- **Fallback mapper:** checking the camera path for jumps and falling back from the global to the incremental mapper when the path breaks.
- **IMU checks:** using the gyro, accelerometer and magnetometer to cross-check the poses.
- **Hole filling:** filling blank walls and ceilings with monocular depth fused into a TSDF.
- **Mesh cleanup:** carving away invented surface and snapping floors, walls and ceilings to planes.
- **OpenMVS:** patching texture bugs locally, and later moving to OpenMVS `develop`, where upstream had fixed them along with a texturing hang.

Some ideas didn't pan out and were dropped, which is part of the fun.

## Where it could go next

- **More data sources:** headsets like Meta Quest, with their cameras, depth and tracking.
- **A capture app:** a dedicated phone app with synced video + IMU and live coverage hints.
- **Multiple captures:** combining several capture sequences into one model.
- **Newer methods at each step:** learned matchers for SfM, learned multi-view depth, and Gaussian splatting or neural surfaces for the final model.
- **Longer captures:** chunked mapping and visual-inertial bundle adjustment.
