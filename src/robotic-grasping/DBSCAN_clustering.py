#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        # ROS topics for depth and RGB images
        self.depth_image_topic = "/camera/depth/image_raw"
        self.rgb_image_topic = "/camera/color/image_raw"

        # Subscribe to topics
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_callback)

        self.depth_image = None
        self.rgb_image = None

        # Camera parameters (example values, replace with actual camera parameters)
        self.fx = 525.0  # Focal length in x
        self.fy = 525.0  # Focal length in y
        self.cx = 319.5  # Principal point in x
        self.cy = 239.5  # Principal point in y

    def depth_callback(self, data):
        try:
            # Convert depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {str(e)}")

    def rgb_callback(self, data):
        try:
            # Convert RGB image to OpenCV format
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")

    def process_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Get image dimensions
            height, width = self.depth_image.shape

            # Create a point cloud from depth and RGB data
            point_cloud = []
            for v in range(height):
                for u in range(width):
                    Z = self.depth_image[v, u]
                    if Z == 0:  # Ignore invalid depth values
                        continue
                    X = (u - self.cx) * Z / self.fx
                    Y = (v - self.cy) * Z / self.fy
                    # Get corresponding RGB value
                    R, G, B = self.rgb_image[v, u]
                    point_cloud.append([X, Y, Z, R, G, B])

            point_cloud = np.array(point_cloud)

            # Perform DBSCAN clustering (tune eps and min_samples)
            clustering = DBSCAN(eps=0.05, min_samples=10).fit(point_cloud[:, :3])  # Use X, Y, Z for clustering

            # Compute density per cluster
            labels = clustering.labels_
            unique_labels = np.unique(labels)
            densities = {}

            for label in unique_labels:
                if label == -1:  # Ignore noise points
                    continue
                cluster_points = point_cloud[labels == label]
                # Compute density as number of points per cluster
                density = len(cluster_points) / (cluster_points[:, :3].ptp(axis=0).prod())  # Volume in 3D space
                densities[label] = density

            # Visualize cluster density as a heatmap (optional)
            heatmap = np.zeros((height, width), dtype=np.float32)
            for v in range(height):
                for u in range(width):
                    Z = self.depth_image[v, u]
                    if Z == 0:
                        continue
                    X = (u - self.cx) * Z / self.fx
                    Y = (v - self.cy) * Z / self.fy
                    R, G, B = self.rgb_image[v, u]
                    point_data = np.array([X, Y, Z, R, G, B])
                    label = clustering.predict([point_data[:, :3]])[0]
                    if label != -1:
                        heatmap[v, u] = densities[label]

            # Normalize heatmap for display
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)

            return heatmap

    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Process the images and calculate cluster density
            heatmap = self.process_images()

            # Display the images
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)
            if heatmap is not None:
                cv2.imshow("Cluster Density Heatmap", heatmap)

            cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.display_images()
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
