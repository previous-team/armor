#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

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
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")

    def calculate_pixel_clutter_density(self, window_size=40):
        if self.rgb_image is None or self.depth_image is None:
            return None
        
        # Converts the RGB image to grayscale, blurs it, and then detects edges using the Canny edge detection
        gray_rgb = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        blurred_rgb = cv2.GaussianBlur(gray_rgb, (5, 5), 0)
        edges = cv2.Canny(blurred_rgb, 50, 150)

        # Initialize clutter density map
        clutter_density_map = np.zeros_like(gray_rgb, dtype=np.float32) #same dimensions as the gray_rgb
        
        
        
        half_window = window_size // 2 #half_window ensures the sliding window is properly centered on each pixel
        
        #Analyzing the window(local) region for clutter
        for y in range(half_window, gray_rgb.shape[0] - half_window):
            for x in range(half_window, gray_rgb.shape[1] - half_window):
                
                # Extracting window around the current pixel
                
                rgb_window = gray_rgb[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                edge_window = edges[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                depth_window = self.depth_image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]

                # Find contours within the window
                contours, _ = cv2.findContours(edge_window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                object_positions = []

                local_clutter_density = 0
                window_area = window_size * window_size

                for contour in contours:
                    # Calculate the bounding box of each object within the window
                    x_w, y_w, w, h = cv2.boundingRect(contour)
                    centroid = (x_w + w // 2, y_w + h // 2)
                    
                    # Average depth of the object
                    depth_region = depth_window[y_w:y_w+h, x_w:x_w+w]
                    avg_depth = np.mean(depth_region)
                    
                    object_positions.append((centroid, avg_depth))
                    object_size = w * h
                    
                    # Calculate proximity factor and contribution to local clutter density
                    for other_centroid, other_depth in object_positions:
                        if other_centroid == centroid:
                            continue

                        # Calculate 2D distance between centroids
                        distance = np.linalg.norm(np.array(centroid) - np.array(other_centroid))
                        
                        # Depth-based proximity factor
                        if abs(avg_depth - other_depth) < 10:  #Takes care of stacked object
                            local_clutter_density += (1 / (distance + 1e-5)) * object_size# # Avoid division by zero
                            #Size of the object (bounding box area) is multiplied by the proximity factor. Larger objects contribute more to clutter density.
                            #Ensures that closer objects contribute more to clutter density. The smaller the distance, the larger the term, which means objects that are closer in 2D space increase the clutter density more significantly

                # Normalize clutter density by window area
                #clutter_density_map[y, x] = local_clutter_density / window_area
                clutter_density_map[y, x] = local_clutter_density 
        return clutter_density_map, edges

    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display the images
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)
            
            # Calculate clutter density map for each pixel
            clutter_density_map, edges = self.calculate_pixel_clutter_density()
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
