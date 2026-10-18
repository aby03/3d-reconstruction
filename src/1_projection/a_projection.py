import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

from src.data_loaders.lego_dataset import LegoDataset

# 1. Load dataset metadata to get camera pose and focal length
data_dir = Path("./data/lego")
with open(data_dir / "transforms_train.json", "r") as f:
    meta = json.load(f)

# Extract field of view to compute focal length (for 400x400 image size)
camera_angle_x = meta["camera_angle_x"]     # horizontal FOV in radians
image_size = 400                            # image size in pixels
fx = fy = (image_size / 2.0) / np.tan(camera_angle_x / 2.0)
cx = cy = image_size / 2.0

# Define Intrinsic Matrix (K)
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1 ]
])

# 2. Pick the first frame's image and its camera-to-world pose
data = LegoDataset(max_files=2)
img_rgb = data.image_tensors[0]
frame = data.frames[0]
c2w = np.array(frame["transform_matrix"], dtype=np.float32)

# 3. Define a 3D point in world coordinates (X, Y, Z)
# The Lego object is roughly centered around [0, 0, 0] in world space
point_world = np.array([0.0, 0.0, 0.0, 1.0])  # Homogeneous 4D vector

# 4. Compute World-to-Camera Matrix (Extrinsics)
# Since transform_matrix is Camera-to-World (C2W), we invert it to get World-to-Camera (W2C)
w2c = np.linalg.inv(c2w)
R = w2c[:3, :3]
T = w2c[:3, 3:4]
extrinsic_matrix = np.hstack((R, T))  # 3x4 matrix

# 5. Project 3D world point to camera coordinates, then to pixels
point_camera = extrinsic_matrix @ point_world  # [X_c, Y_c, Z_c]^T
pixel_homogeneous = K @ point_camera

# Perspective division by depth (Z_c)
u = pixel_homogeneous[0] / pixel_homogeneous[2]
v = pixel_homogeneous[1] / pixel_homogeneous[2]

print(f"Projected Pixel Coordinates: u = {u:.2f}, v = {v:.2f}")

# 6. Visualize using Matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
# Plot the projected 3D point as a red dot
plt.scatter([u], [v], color='red', s=100, marker='o', label='Projected 3D Point (0,0,0)')
plt.xlim(0, image_size)
plt.ylim(image_size, 0)  # Flip y-axis to match image coordinate standards
plt.legend()
plt.title("3D Point Projection Visualization")
plt.axis('off')
plt.show()
