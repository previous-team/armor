#!/usr/bin/env python3

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.cam_gazebo import ROSCameraSubscriber
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data_gazebo import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import Grasp,detect_grasps

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import peak_local_max
from utils.dataset_processing import image

logging.basicConfig(level=logging.INFO)

def get_z(img):
    depth_img = image.Image(img)
    depth_img.resize((480,640))

    width=640
    height=480
    output_size=224

    left = (width - output_size) // 2
    top = (height - output_size) // 2
    right = (width + output_size) // 2
    bottom = (height + output_size) // 2

    bottom_right = (bottom, right)
    top_left = (top, left)
    depth_img.crop(bottom_right=bottom_right, top_left=top_left)

    return depth_img.img

def z_detect_grasps(depth,q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        cx,cy=grasp_point

        print(f"Grasp point (cx, cy): ({cx}, {cy})")

        depth_img=get_z(depth)
        z=depth_img[cx,cy]
        print("z:",z)


        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasp_point,z


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
           
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                grasps=z_detect_grasps(depth,q_img, ang_img, width_img=None, no_grasps=1)
                print(grasps)


                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)
                
    finally:
        pass
