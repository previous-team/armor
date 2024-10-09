#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import open3d as o3d

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        self.depth_image_topic = "/camera/depth/image_raw"
        self.rgb_image_topic = "/camera/color/image_raw"
        
        # Subscribers
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_callback)

        # Placeholder for images
        self.depth_image = None
        self.rgb_image = None

        # Camera intrinsics (example values for RealSense D435)
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=462.1379699707031, fy=462.1379699707031, cx=320.0, cy=240.0
        )

    def depth_callback(self, data):
        try:
            # Convert ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            # Normalize for display
            self.depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {str(e)}")

    def rgb_callback(self, data):
        try:
            # Convert ROS RGB image to OpenCV format
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")

    def process_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Convert OpenCV images to Open3D images
            depth_o3d = o3d.geometry.Image(self.depth_image)
            rgb_o3d = o3d.geometry.Image(self.rgb_image)

            # Create an RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(self.rgb_image),
                o3d.geometry.Image(self.depth_image),
                depth_scale=1000.0, convert_rgb_to_intensity=False)

            # Create point cloud from the RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsics)

            # Downsample the point cloud to reduce noise
            pcd = pcd.voxel_down_sample(voxel_size=0.02)

            # Estimate the normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Segment the largest plane (e.g., floor or table) using RANSAC
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers, invert=True)

            # Use DBSCAN to find clusters in the outlier points (clutter objects)
            labels = np.array(outlier_cloud.cluster_dbscan(eps=0.05, min_points=10, print_progress=False))

            # Count the number of clusters detected (ignoring noise)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Number of clusters detected: {num_clusters}")

            # Calculate density by checking the number of points per cluster
            for cluster_id in range(num_clusters):
                cluster_points = np.where(labels == cluster_id)[0]
                print(f"Cluster {cluster_id}: Number of points = {len(cluster_points)}")

            # Visualize the result (optional)
            o3d.visualization.draw_geometries([outlier_cloud], window_name="Clutter Detection")

    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display depth and RGB images using OpenCV
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)
            cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.display_images()
            self.process_images()  # Add the processing of point cloud and clustering
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('camera_subscriber', anonymous=True)
    camera_subscriber = CameraSubscriber()
    try:
        camera_subscriber.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
