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
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)  # Calculate gradient in x direction
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)  # Calculate gradient in y direction

        # Magnitude of the gradient
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        density_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return density_map

    def calculate_clutter_density(self):
        if self.rgb_image is None or self.depth_image is None:
            return None
        
        # Calculate density from gradients
        gradient_density = self.compute_density_from_gradient(self.depth_image)

        # Convert the RGB image to grayscale and apply edge detection
        gray_rgb = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        blurred_rgb = cv2.GaussianBlur(gray_rgb, (5, 5), 0)
        edges = cv2.Canny(blurred_rgb, 50, 150)

        # Find contours in the image to detect objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Total image area for normalization
        total_area = self.rgb_image.shape[0] * self.rgb_image.shape[1]  
        clutter_density = np.zeros_like(self.depth_image, dtype=np.float32)

        for contour in contours:
            # Calculate the bounding box of each object
            x, y, w, h = cv2.boundingRect(contour)
            depth_region = self.depth_image[y:y+h, x:x+w]
            avg_depth = np.mean(depth_region)

            # Calculate object size (approximated by bounding box area)
            object_size = w * h

            # For each object, compute clutter contribution based on proximity to others
            for other_contour in contours:
                if other_contour is contour:  # Skip self
                    continue

                other_x, other_y, other_w, other_h = cv2.boundingRect(other_contour)
                other_centroid = (other_x + other_w // 2, other_y + other_h // 2)

                # Calculate 2D distance between centroids
                distance = np.linalg.norm(np.array((x + w // 2, y + h // 2)) - np.array(other_centroid))

                # Depth-based proximity factor
                other_depth_region = self.depth_image[other_y:other_y + other_h, other_x:other_x + other_w]
                other_avg_depth = np.mean(other_depth_region)

                if abs(avg_depth - other_avg_depth) < 50:  # Adjust threshold as needed
                    clutter_density[y:y+h, x:x+w] += (1 / (distance + 1e-5)) * object_size  # Avoid division by zero

        # Combine the gradient density with the clutter density
        combined_density = cv2.addWeighted(clutter_density, 0.5, gradient_density.astype(np.float32), 0.5, 0)
        
        # Normalize combined density by maximum value
        combined_density /= np.max(combined_density) if np.max(combined_density) > 0 else 1

        return combined_density, contours, edges

    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display the images
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)

            # Calculate clutter density and display the result
            combined_density, contours, edges = self.calculate_clutter_density()
            if combined_density is not None:
                print(f"Combined Clutter Density computed.")

                # Apply heatmap to combined density
                combined_density_heatmap = cv2.applyColorMap((combined_density * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Display combined density as a heatmap
                cv2.imshow("Combined Clutter Density Heatmap", combined_density_heatmap)

            cv2.imshow("Canny Edges", edges)

            # Draw contours on the RGB image
            rgb_with_contours = self.rgb_image.copy()
            cv2.drawContours(rgb_with_contours, contours, -1, (0, 255, 0), 2)  # Draw green contours

            # Show the RGB image with contours
            cv2.imshow("RGB with Contours", rgb_with_contours)

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
