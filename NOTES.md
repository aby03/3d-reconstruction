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

## From dense reconstruction to Gaussian splatting

**Where dense reconstruction stops.** Patch matching needs texture and a clear view, so blank walls, surfaces seen at
a steep angle and hidden corners stay empty. Filling them with monocular depth helped, but the patches are guesses
glued on, and a learned multi-view depth model didn't do better. The holes come from how the surface is built, not
from the depth source.

**Why splatting.** Gaussian splatting fits the whole scene to all the photos at once, so a surface seen poorly from
many views still gets filled. 3DGS looks best, but its blobs have no clear surface; 2DGS uses flat discs, which give
a real surface to build a mesh from. Its discs also have a clear depth, so depth readings from future sensors
(a headset's depth camera, a phone's LiDAR) could guide them directly. That made 2DGS the better fit.

**How it was tried.** As a second option next to OpenMVS rather than a replacement, so the two can be compared on the
same cameras. Kept to roughly the same time budget and to a laptop GPU, with training that can stop and continue.
Monocular depth guides the blank walls, some photos are held out to judge honestly, and the mesh comes from the
rendered depth so the rest of the pipeline stays the same.

**What it showed.** The holes mostly close, and the splats render convincing views from angles no one filmed. The
trade-off is a smoother mesh with less fine detail, and areas no photo saw well get the right shape but no colour.

## Where it could go next

- **More data sources:** headsets like Meta Quest, with their cameras, depth and tracking.
- **A capture app:** a dedicated phone app with synced video + IMU and live coverage hints.
- **Multiple captures:** combining several capture sequences into one model.
- **Newer methods at each step:** learned matchers for SfM, sharper meshes from splats, and colouring what the photos never saw.
- **Longer captures:** chunked mapping and visual-inertial bundle adjustment.
