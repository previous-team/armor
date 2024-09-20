import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2.aruco as aruco
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results

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

def create_target_mask(rgb_image):
    # TODO its only for red object here maybe some other idea can be incorporated depending on how well the model predicts for masked image
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 100, 100])
    upper_red_2 = np.array([179, 255, 255])
    mask_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    return red_mask

def apply_mask_to_image(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')

    # Get the compute device
    device = get_device(args.force_cpu)

    #net = torch.load(args.network)
    net = torch.load(args.network)
    #net = net.to(device)
    logging.info('Done')

    

    try:
        fig = plt.figure(figsize=(10, 10))
        while True:
            image_bundle = cam.get_image_bundle()

            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            #mask for the target object(here red object)
            red_mask = create_target_mask(rgb)

            #mask to both RGB and depth images
            masked_rgb = apply_mask_to_image(rgb, red_mask)
            masked_depth = apply_mask_to_image(depth, red_mask)
            masked_depth = np.expand_dims(masked_depth, axis=2)
            
            print(f'masked depth\n{masked_depth.shape}')

            x, depth_img, rgb_img = cam_data.get_data(rgb=masked_rgb, depth=masked_depth)
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)
    finally:
        save_results(
            rgb_img=cam_data.get_rgb(rgb, False),
            depth_img=np.squeeze(cam_data.get_depth(depth)),
            grasp_q_img=q_img,
            grasp_angle_img=ang_img,
            no_grasps=args.n_grasps,
            grasp_width_img=width_img
        )
