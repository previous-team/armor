#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class CameraProcessor:
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
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
            self.depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {str(e)}")

    def rgb_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")

    def process_images(self):
        if self.rgb_image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
            # Apply a blur to reduce noise before edge detection
            gray = cv2.medianBlur(gray, 5)

            centers = []

            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1,  # Inverse ratio of resolution
                minDist=100,  # Minimum distance between detected centers
                param1=50,  # Upper threshold for the internal Canny edge detector
                param2=20,  # Threshold for center detection
                minRadius=1,  # Minimum radius to be detected
                maxRadius=15  # Maximum radius to be detected
            )

            # If some circles are detected, draw them
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                print(circles)
                for (x, y, r) in circles:
                    # Draw the circle in the output image
                    cv2.circle(self.rgb_image, (x, y), r, (0, 255, 0), 4)
                    # Draw a rectangle at the center of the circle
                    cv2.rectangle(self.rgb_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    centers.append((x, y))

                if centers:
                    # Compute the bounding box
                    x_coords, y_coords = zip(*centers)
                    x_min, x_max = max(0, min(x_coords)), min(self.rgb_image.shape[1], max(x_coords))
                    y_min, y_max = max(0, min(y_coords)), min(self.rgb_image.shape[0], max(y_coords))
                    
                    # Draw the bounding rectangle
                    cv2.rectangle(self.rgb_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    # Crop the image to the bounding box
                    if x_min < x_max and y_min < y_max:
                        cropped_image = self.rgb_image[y_min:y_max, x_min:x_max]

                        # Display the cropped image if it's valid
                        if cropped_image.size > 0:
                            cv2.imshow("Cropped Image", cropped_image)

            # Display the resulting image
            cv2.imshow("Processed RGB Image", self.rgb_image)
            
        if self.depth_image is not None:
            # Display the depth image
            cv2.imshow("Depth Image", self.depth_image)

        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.process_images()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('camera_processor', anonymous=True)
    camera_processor = CameraProcessor()
    try:
        camera_processor.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
