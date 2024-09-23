#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ROSCameraSubscriber:
    def __init__(self, depth_topic='/camera/depth/image_raw', rgb_topic='/camera/color/image_raw'):
        self.bridge = CvBridge()
        self.depth_topic = depth_topic
        self.rgb_topic = rgb_topic

        self.depth_image = None
        self.rgb_image = None

        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, '32FC1')  
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")

    

    def rgb_callback(self, data):
        try:
           
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8") # Convert ROS RGB image to a numpy array
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {str(e)}")

    def get_image_bundle(self):
        if self.rgb_image is None or self.depth_image is None:
            return None  # Images are not yet available

        depth_image = np.expand_dims(self.depth_image, axis=2)
        return {
            'rgb': self.rgb_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()
        if images is None:
            rospy.loginfo("Waiting for images...")
            return

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('RGB Image')
        ax[0, 1].set_title('Aligned Depth Image')

        plt.show()

if __name__ == '__main__':
    rospy.init_node('gazebo_depth_camera', anonymous=True)
    cam = ROSCameraSubscriber(
        depth_topic='/camera/depth/image_raw',  
        rgb_topic='/camera/color/image_raw'    
    )
    try:
        while not rospy.is_shutdown():
            cam.plot_image_bundle()
            rospy.sleep(0.1)  
    except rospy.ROSInterruptException:
        pass
