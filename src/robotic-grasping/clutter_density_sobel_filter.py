#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        # Topics for RGB and depth images
        self.depth_image_topic = "/camera/depth/image_raw"
        self.rgb_image_topic = "/camera/color/image_raw"

        # Subscribers for RGB and depth images
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_callback)

        self.depth_image = None
        self.rgb_image = None

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

    def compute_density_from_gradient(self, depth_image):
        """Compute the 'density' based on depth image gradients."""
        # Apply Sobel filters to compute depth gradients
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=31)#calcualtes gradient in x direction
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=31) #calculates gradient in y direction
        

        # Magnitude of the gradient
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        print(gradient_magnitude)


        density = cv2.normalize(gradient_magnitude , None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        print("Density:",0)

        return density

    def apply_heatmap(self, density_map):
        """Apply a heatmap (color map) to the density map."""
        # Apply a heatmap color mapping to the density map
        heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
        return heatmap

    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display the original RGB and depth images
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)

            # Compute cluster density from depth gradient
            density_map = self.compute_density_from_gradient(self.depth_image)

            # Apply heatmap to the density map
            heatmap = self.apply_heatmap(density_map)

            # Display the heatmap
            cv2.imshow("Density Heatmap", heatmap)

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
