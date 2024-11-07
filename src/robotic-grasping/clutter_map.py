#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import concurrent.futures
from skimage.transform import resize

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        self.depth_image_topic = "/camera/depth/image_raw"
        self.rgb_image_topic = "/camera/color/image_raw"
        
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_callback)

        self.depth_image = None
        self.rgb_image = None

    def crop(self, img, top_left, bottom_right):
    
    # Crop the image to a bounding box given by top left and bottom right pixels.
    # :param top_left: tuple, top left pixel.
    # :param bottom_right: tuple, bottom right pixel
    # :param resize: If specified, resize the cropped image to this size
    
        dept = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        return dept
    
    def resize(self, img, shape):
        """
        Resize image to shape.
        :param shape: New shape.
        """
        if img.shape == shape:
            return
        return resize(img, shape, preserve_range=True).astype(img.dtype)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.depth_image = self.crop(self.depth_image, top_left=(0, 160), bottom_right=(720, 1120))
            self.depth_image = resize(self.depth_image, (480, 640)) #for software only
            self.depth_image = self.crop(self.depth_image, bottom_right=(352,432), top_left=(128,208))
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
        # gray = self.depth_image.copy()
        # gray = np.uint8(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 200)

        # Find contours in the image to detect objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Calculate distances between object centroids and their sizes
        objects = []
        total_area = self.rgb_image.shape[0] * self.rgb_image.shape[1]  # Total image area

        self.test_img = self.rgb_image.copy()

        for contour in contours:
            # Calculate the bounding box of each object
            x, y, w, h = cv2.boundingRect(contour)
            
            # Remove small objects (noise)
            if w * h < 0.01 * total_area:
                continue
            else:
                cv2.rectangle(self.test_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

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

            # Display test image
            cv2.imshow("Test image", self.test_img)

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
