import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data_loaders.lego_dataset import LegoDataset
from src.feature_matching.sift_matcher import SIFTMatcher

# 1. Load data
data = LegoDataset(max_files=6)
img1_index = 0
img2_index = 1

img1 = data.image_tensors[img1_index].numpy()
img2 = data.image_tensors[img2_index].numpy()

# Intrinsics for the resized images. Translation recovered from two views has
# unknown scale when no pose or scene-scale information is provided.
camera_fov_radians = data.camera_angle_x
image_size = data.image_size
fx = fy = (image_size / 2.0) / np.tan(camera_fov_radians / 2.0)
cx = cy = image_size / 2.0
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# 2. Keypoints extraction and matching
matcher = SIFTMatcher()
pts1, pts2, good_matches, scores, keypoints1, keypoints2 = matcher.match(img1, img2)
if len(good_matches) < 5:
    raise RuntimeError(f"Only {len(good_matches)} SIFT matches found; need at least 5")

img1_display = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
img2_display = np.clip(img2 * 255.0, 0, 255).astype(np.uint8)
all_matches_image = cv2.drawMatches(
    img1_display,
    keypoints1,
    img2_display,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

if pts1.shape[0] != pts2.shape[0]:
    raise RuntimeError("Feature matcher returned different numbers of points per image")

# 3. Estimate relative pose from intrinsics and matched pixels
essential_matrix, ransac_mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    cv2.RANSAC,
    0.999,
    1.0,
)
if essential_matrix is None or ransac_mask is None:
    raise RuntimeError("Essential-matrix estimation failed for these image matches")

ransac_mask = ransac_mask.ravel().astype(bool)
if np.count_nonzero(ransac_mask) < 5:
    raise RuntimeError("Essential-matrix estimation found fewer than 5 inlier matches")

_, rotation, translation, pose_mask = cv2.recoverPose(
    essential_matrix,
    pts1,
    pts2,
    K,
    mask=ransac_mask.astype(np.uint8),
)
pose_inliers = pose_mask.ravel().astype(bool)
if np.count_nonzero(pose_inliers) < 2:
    raise RuntimeError("Pose recovery found too few cheirality-consistent matches")

# 4. Triangulate pose inliers in camera 1's coordinate frame
pose_pts1 = pts1[pose_inliers]
pose_pts2 = pts2[pose_inliers]
projection1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
projection2 = K @ np.hstack((rotation, translation))
points_h = cv2.triangulatePoints(projection1, projection2, pose_pts1.T, pose_pts2.T)

valid_homogeneous = np.abs(points_h[3]) > 1e-8
points_3d = (points_h[:3, valid_homogeneous] / points_h[3, valid_homogeneous]).T
points_3d_cam2 = (rotation @ points_3d.T + translation).T
positive_depth = (points_3d[:, 2] > 0) & (points_3d_cam2[:, 2] > 0)
finite_points = np.isfinite(points_3d).all(axis=1)
points_3d = points_3d[positive_depth & finite_points]
if len(points_3d) == 0:
    raise RuntimeError("Triangulation produced no finite points in front of both cameras")

filtered_matches_image = cv2.drawMatches(
    img1_display,
    keypoints1,
    img2_display,
    keypoints2,
    good_matches,
    None,
    matchesMask=ransac_mask.astype(np.uint8).tolist(),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].imshow(all_matches_image)
axes[0].set_title(f"Ratio-test matches: {len(good_matches)}")
axes[0].axis("off")
axes[1].imshow(filtered_matches_image)
axes[1].set_title(f"Essential-matrix inliers: {np.count_nonzero(ransac_mask)}")
axes[1].axis("off")
fig.tight_layout()

fig_3d = plt.figure(figsize=(9, 8))
axis_3d = fig_3d.add_subplot(111, projection="3d")
scatter = axis_3d.scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    points_3d[:, 2],
    c=points_3d[:, 2],
    cmap="viridis",
    s=8,
)
camera2_center = (-rotation.T @ translation).ravel()
axis_3d.scatter(*camera2_center, color="red", marker="^", s=70, label="Camera 2")
axis_3d.scatter(0, 0, 0, color="black", marker="o", s=50, label="Camera 1")
axis_3d.set_xlabel("X (camera 1 frame)")
axis_3d.set_ylabel("Y (camera 1 frame)")
axis_3d.set_zlabel("Z (camera 1 frame)")
axis_3d.set_title(f"Triangulated points: {len(points_3d)} (scale is arbitrary)")
axis_3d.legend()
fig_3d.colorbar(scatter, ax=axis_3d, label="Depth in camera 1 frame")

output_dir = Path(__file__).resolve().parents[3] / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
matches_path = output_dir / "tutorial_1c_matches.png"
point_cloud_path = output_dir / "tutorial_1c_point_cloud.png"
fig.savefig(matches_path, dpi=160, bbox_inches="tight")
fig_3d.savefig(point_cloud_path, dpi=160, bbox_inches="tight")

print(f"SIFT matches: {len(good_matches)}")
print(f"Essential-matrix inliers: {np.count_nonzero(ransac_mask)}")
print(f"Pose inliers: {np.count_nonzero(pose_inliers)}")
print(f"Finite points in front of both cameras: {len(points_3d)}")
print("Relative rotation (camera 1 to camera 2):")
print(rotation)
print("Relative translation direction (scale is unknown):")
print(translation.ravel())
print(f"Matches visualization saved to: {matches_path}")
print(f"Point-cloud visualization saved to: {point_cloud_path}")

plt.show()
