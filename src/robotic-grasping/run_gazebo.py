#!/usr/bin/env python3

import argparse
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from tf.transformations import quaternion_from_euler

from hardware.cam_gazebo import ROSCameraSubscriber
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data_gazebo import CameraData
from utils.visualisation.plot import plot_results
from utils.dataset_processing.grasp import Grasp

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import peak_local_max
from utils.dataset_processing import image

logging.basicConfig(level=logging.INFO)


def deproject_pixel_to_point(depth, pixel, ppx, ppy, fx, fy):
    x = (pixel[0] - ppx) * depth / fx
    y = (pixel[1] - ppy) * depth / fy
    return x, y, depth

def filter_grasps(grasps, img, red_thresh=0, green_thresh=-1, blue_thresh=-1):
    """
    Filter out grasps that are not on the object
    :param grasps: list of Grasps
    :param img: RGB Image # Shape: (3, 224, 224)
    :param red_thresh: Red threshold
    :param green_thresh: Green threshold
    :param blue_thresh: Blue threshold
    :return: list of Grasps
    """
    filtered_grasps = []
    for g in grasps:
        cy, cx = g.center
        print(img[0, cy, cx])
        if img[0, cy, cx] > red_thresh and img[1, cy, cx] > green_thresh and img[2, cy, cx] > blue_thresh:
            filtered_grasps.append(g)

    return filtered_grasps

def z_detect_grasps(rgb_img, depth, q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=10, threshold_abs=0.1, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp = Grasp(grasp_point, ang_img[grasp_point])
        if width_img is not None:
            grasp.length = width_img[grasp_point]
            grasp.width = grasp.length / 2

        grasps.append(grasp)

    filtered_grasps = filter_grasps(grasps, rgb_img)

    for g in filtered_grasps:
        print("Grasp at: ", g.center, g.angle)

        cy,cx = g.center  # Invert to reverse the transpose operation that is given to the network

        g.angle = g.angle

        print("Grasp angle: ", math.degrees(g.angle))
        
        quaternion = quaternion_from_euler(0, 0, g.angle) #rotation about the z-axis

        z = depth[cy, cx]

        fx = 462.1379699707031
        fy = 462.1379699707031
        ppx = 111
        ppy = 111

        x, y, z = deproject_pixel_to_point(z, (cx, cy), ppx, ppy, fx, fy)

        print("Grasp at: ", x, y, z)

        # Publish the grasp point
        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = rospy.Time.now()
        grasp_msg.pose.position.x = x / 1000
        grasp_msg.pose.position.y = y / 1000
        grasp_msg.pose.position.z = z / 1000
        
        # Orientation will be published later
        grasp_msg.pose.orientation.x = quaternion[0]
        grasp_msg.pose.orientation.y = quaternion[1]
        grasp_msg.pose.orientation.z = quaternion[2]
        grasp_msg.pose.orientation.w = quaternion[3]

        grasp_pub.publish(grasp_msg)

    return filtered_grasps


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='src/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    rospy.init_node('run_gazebo', anonymous=True)

    cam = ROSCameraSubscriber(
        depth_topic='/camera/depth/image_raw',  
        rgb_topic='/camera/color/image_raw'    
    )

    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Grasp point publisher
    grasp_pub = rospy.Publisher('/grasp_point', PoseStamped, queue_size=10)

    # Load Network
    logging.info('Loading model...')

    if torch.cuda.is_available() and not args.force_cpu:
        net = torch.load(args.network)
    else:
        net = torch.load(args.network, map_location=torch.device('cpu'))

    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    try:
        fig = plt.figure(figsize=(10, 10))
        while True:
            image_bundle = cam.get_image_bundle()
            if image_bundle is None:
                rospy.loginfo("Waiting for images...")
                continue

            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
           
            x, depth_img, denormalised_depth, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                grasps=z_detect_grasps(rgb_img, denormalised_depth, q_img, ang_img, width_img=None, no_grasps=10)


                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(denormalised_depth),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=10,
                             grasp_width_img=width_img)
                
    finally:
        pass
