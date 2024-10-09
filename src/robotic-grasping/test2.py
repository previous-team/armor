import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

# Configure Intel RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Skip a few frames to allow auto-exposure and gain to adjust
for _ in range(5):
    pipeline.wait_for_frames()

# Get frames from the camera
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convert RealSense frames to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Stop the camera
pipeline.stop()

# Convert to Open3D images
depth_o3d = o3d.geometry.Image(depth_image)
color_o3d = o3d.geometry.Image(color_image)

# Create an RGBD image from the depth and color frames
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False)

profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color)
intrinsics = video_stream.as_video_stream_profile().get_intrinsics()

# Define the camera intrinsics for RealSense D435
camera_intrinsics =camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])

# Generate point cloud from the RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

# Downsample the point cloud to reduce noise and make it more manageable
pcd = pcd.voxel_down_sample(voxel_size=0.02)

# Estimate normals for the point cloud
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Segment the largest plane (e.g., table or floor) using RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Visualize segmented point cloud
print("Showing segmented point cloud (clutter)")
o3d.visualization.draw_geometries([outlier_cloud], window_name="Clutter Density Detection")

# Use DBSCAN to find clusters in the outlier points (clutter objects)
labels = np.array(outlier_cloud.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

# Count the number of clusters detected (ignoring noise)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters detected: {num_clusters}")

# Calculate density by checking the number of points per cluster
for cluster_id in range(num_clusters):
    cluster_points = np.where(labels == cluster_id)[0]
    print(f"Cluster {cluster_id}: Number of points = {len(cluster_points)}")

# Use cluster information as input for the RL algorithm (optional)
