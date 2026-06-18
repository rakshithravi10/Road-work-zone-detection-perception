import cv2
import numpy as np
import open3d as o3d

# ==============================
# PATHS (cam1, frame 000000)
# ==============================

IMAGE_PATH = "../extrinsic/cam1/rosbag2_2025_10_21-16_51_52/image_1/000000.png"
LIDAR_PATH = "../extrinsic/cam1/rosbag2_2025_10_21-16_51_52/lidar_bin_1/000000.bin"

# Load image
img = cv2.imread(IMAGE_PATH)
img_vis = img.copy()

# Load LiDAR
points = np.fromfile(LIDAR_PATH, dtype=np.float32).reshape(-1, 4)
points_xyz = points[:, :3]

# Visualize LiDAR
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_xyz)
o3d.visualization.draw_geometries([pcd])

cv2.imshow("Image", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
