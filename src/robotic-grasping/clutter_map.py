#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import concurrent.futures

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        self.depth_image_topic = "/camera/depth/image_raw"
        self.rgb_image_topic = "/camera/color/image_raw"
        
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_callback)

        self.depth_image = None
        self.rgb_image = None

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {str(e)}")

    def rgb_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self.rgb_image = self.rgb_image[128:352, 208:432]
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")
            
    def calculate_clutter_density(self):
        if self.rgb_image is None or self.depth_image is None:
            return None
        
        # Convert the RGB image to grayscale and apply edge detection
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the image to detect objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate distances between object centroids and their sizes
        objects = []
        total_area = self.rgb_image.shape[0] * self.rgb_image.shape[1]  # Total image area

        for contour in contours:
            # Calculate the bounding box of each object
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w // 2, y + h // 2)

            # Find the corresponding depth of the object by averaging depth pixels within the bounding box
            depth_region = self.depth_image[y:y+h, x:x+w]
            avg_depth = np.mean(depth_region)

            # Calculate object size (approximated by bounding box area)
            object_size = w * h
            
            # Append object position and depth
            objects.append((centroid, avg_depth, object_size))
        
        # Initialize clutter density map
        clutter_density_map = np.zeros_like(gray, dtype=np.float32)

        # Define window size
        window_size = 5  # Adjust this value as needed

        def calculate_clutter_for_window(x, y):
            clutter_density = 0
            for other_centroid, other_depth, object_size in objects:
                distance = np.linalg.norm(np.array((x, y)) - np.array(other_centroid))
                clutter_density += (1 / (distance + 1e-5)) * object_size
            return clutter_density

        for x in range(0, self.rgb_image.shape[1], window_size):
            for y in range(0, self.rgb_image.shape[0], window_size):
                clutter_density = min(calculate_clutter_for_window(x, y), 500)
                clutter_density_map[y:y+window_size, x:x+window_size] = clutter_density

        return clutter_density_map, edges
    
    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display the images
            # cv2.imshow("Depth Image", self.depth_image)
            # cv2.imshow("RGB Image", self.rgb_image)
            
            # Calculate clutter density map for each pixel
            clutter_density_map, edges = self.calculate_clutter_density()
            print("clutter:",clutter_density_map)
            print("length:",len(clutter_density_map))
            if clutter_density_map is not None:
                print("Clutter density map calculated.")
            
            # Normalize the clutter density map for heatmap display
            clutter_density_normalized = cv2.normalize(clutter_density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply a color map to create a heatmap
            heatmap = cv2.applyColorMap(clutter_density_normalized, cv2.COLORMAP_JET)

            # Display the heatmap
            cv2.imshow("Clutter Density Heatmap", heatmap)
            
            # Display the Canny edges (optional)
            cv2.imshow("Canny Edges", edges)

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
