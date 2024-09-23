#!/usr/bin/env python3

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.cam_gazebo import ROSCameraSubscriber
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np

import cv2  

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='/home/sanraj/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
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
    


def isolate_red_objects(rgb_image):

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    mask = mask1 | mask2

    red_only_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

    return red_only_image   


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

    # Load Network
    logging.info('Loading model...')

    if torch.cuda.is_available() and not args.force_cpu:
        net = torch.load(args.network)
    else:
        net = torch.load(args.network, map_location=torch.device('cpu'))

    #net = torch.load(args.network)
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


            red_only_rgb = isolate_red_objects(rgb)

            # Process the red-only image for grasp detection
            x, depth_img, rgb_img = cam_data.get_data(rgb=red_only_rgb, depth=depth)
            
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                print("pred:",pred)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)

                # OpenCV Plot
                rgb_bgr = cv2.cvtColor(cam_data.get_rgb(rgb, False), cv2.COLOR_RGB2BGR)
                depth_img = np.squeeze(cam_data.get_depth(depth))
                q_img = np.squeeze(q_img)
                ang_img = np.squeeze(ang_img)
                width_img = np.squeeze(width_img)

                cv2.imshow('RGB Image', rgb_bgr)
                cv2.imshow('Depth Image', depth_img)
                cv2.imshow('Grasp Quality Image', q_img)
                cv2.imshow('Grasp Angle Image', ang_img)
                cv2.imshow('Grasp Width Image', width_img)

                # Break loop on key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        save_results(
            rgb_img=cam_data.get_rgb(rgb, False),
            depth_img=np.squeeze(cam_data.get_depth(depth)),
            grasp_q_img=q_img,
            grasp_angle_img=ang_img,
            no_grasps=args.n_grasps,
            grasp_width_img=width_img
        )
