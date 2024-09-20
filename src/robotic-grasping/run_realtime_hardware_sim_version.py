import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import Grasp

from skimage.feature import peak_local_max
import pyrealsense2 as rs

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='/home/archanaa/armor/Ned_niryo/src/armor/robotic-grasping-original/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
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


def z_detect_grasps(intrinsics,depth0,depth, q_img, ang_img, width_img=None, no_grasps=1):
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

        cy,cx = grasp_point  # Invert to reverse the transpose operation that is given to the network

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)
        print(intrinsics)
        z = depth0.get_distance(cx, cy)
        

        # fx = 462.1379699707031
        # fy = 462.1379699707031
        ppx = 111
        ppy = 111
        x = (cx - ppx) / intrinsics.fx * z
        y = (cy - ppy) / intrinsics.fy * z

        print("Grasp at: ", x, y, z)


        # Publish the grasp point
        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = rospy.Time.now()
        grasp_msg.pose.position.x = x
        grasp_msg.pose.position.y = y 
        grasp_msg.pose.position.z = z
        
        print(grasp_msg.pose.position.x)
        # Orientation will be published later

        grasp_pub.publish(grasp_msg)

    return grasps


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=830112070066)
    
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
    print(cam.intrinsics)
    intrinsics=cam.intrinsics

    #initialising node
    rospy.init_node('grasp_node', anonymous=True)

    # Grasp point publisher
    grasp_pub = rospy.Publisher('/grasp_point', PoseStamped, queue_size=10)

    # Load Network
    logging.info('Loading model...')

    # Get the compute device
    device = get_device(args.force_cpu)
    #net = torch.load(args.network)
    net = torch.load(args.network)

    logging.info('Done')

    

    try:
        fig = plt.figure(figsize=(10, 10))
        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img,denormalised_depth, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                grasps=z_detect_grasps(intrinsics,depth_img,denormalised_depth, q_img, ang_img, width_img=None, no_grasps=1)
                print(grasps)
                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(denormalised_depth),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)
    finally:
        pass
        # save_results(
        #     rgb_img=cam_data.get_rgb(rgb, False),
        #     depth_img=np.squeeze(cam_data.get_depth(depth)),
        #     grasp_q_img=q_img,
        #     grasp_angle_img=ang_img,
        #     no_grasps=args.n_grasps,
        #     grasp_width_img=width_img
        # )