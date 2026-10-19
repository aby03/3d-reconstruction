import matplotlib.pyplot as plt
import numpy as np

from src.data_loaders.lego_dataset import LegoDataset

# 1. Load data
data = LegoDataset(max_files=6)

camera_fov_radians = data.camera_angle_x
image_size = data.image_size

fx = fy = (image_size / 2.0) / np.tan(camera_fov_radians / 2.0)
cx = cy = image_size / 2.0

# Intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# 2. Select 2 images
img1_index, img2_index = 0, 5
frame1 = data.frames[img1_index]
frame2 = data.frames[img2_index]

c2w_1 = np.array(frame1["transform_matrix"], dtype=np.float32)
c2w_2 = np.array(frame2["transform_matrix"], dtype=np.float32)

img1 = data.image_tensors[img1_index]
img2 = data.image_tensors[img2_index]

# 3. Compute relative pose between frame 1 and frame 2
# Given frame 1 and frame 2 world coordinates frame of reference is same
w2c_1 = np.linalg.inv(c2w_1)
w2c_2 = np.linalg.inv(c2w_2)

R1, T1 = w2c_1[:3, :3], w2c_1[:3, 3:4]
R2, T2 = w2c_2[:3, :3], w2c_2[:3, 3:4]

# Transformation from camera 1 to camera 2
# Following transform any point in camera 1 to camera 2
# X_c2 = R_rel @ X_c1 + T_rel
R_rel = R2 @ R1.T
T_rel = T2 - R_rel @ T1

# 4. Essential and Fundamental Matrices
def skew_symmetric(v):
    return np.array([
        [0, -v[2, 0], v[1, 0]],
        [v[2, 0], 0, -v[0, 0]],
        [-v[1, 0], v[0, 0], 0]
    ])

# Applying coplanarity constraint to:
# X_c2 = R_rel @ X_c1 + T_rel
# We get X_c2 . (T_rel x R_rel @ X_c1) = 0
# Also, A x B = A_skew @ B
# This gives: X_c2 . (T_skew @ R_rel) @ X_c1 = 0
# X_c2_T @ (T_skew @ R_rel) @ X_c1 = 0
# Essential Matrix: T_skew @ R_rel
T_skew = skew_symmetric(T_rel)
E = T_skew @ R_rel

# Fundamental Matrix
K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

# 5. Project 3d world point and draw epipolar lines
point_world = np.array([0.1, 0.1, 0.2, 1.0])

p_cam1 = w2c_1 @ point_world
pix1 = K @ (p_cam1[:3] / p_cam1[2])
u1, v1 = pix1[0] / pix1[2], pix1[1] / pix1[2]

p_cam2 = w2c_2 @ point_world
pix2 = K @ (p_cam2[:3] / p_cam2[2])
u2, v2 = pix2[0] / pix2[2], pix2[1] / pix2[2]

# Draw (u1, v1) epipolar line in image 2
x1 = np.array([u1, v1, 1.0])
l2 = F @ x1

# 6. Plot results
fig, axes = plt.subplots(1, 2)

# Image 1 plot
axes[0].imshow(img1)
axes[0].scatter([u1], [v1], color='cyan', s=80, marker='x', label='Point in View 1')
axes[0].set_title('View 1')
axes[0].legend()

axes[1].imshow(img2)
axes[1].scatter([u2], [v2], color='red', s=80, marker='o', label='Point in View 2')
# Epipolar line
u_vals = np.array([0, image_size])
if abs(l2[1]) > 1e-5:
    v_vals = (-l2[0] * u_vals - l2[2]) / l2[1]
    axes[1].plot(u_vals, v_vals, color='yellow', linewidth=2, label='Epipolar line')
axes[1].set_title("View 2")
axes[1].legend()

plt.show()