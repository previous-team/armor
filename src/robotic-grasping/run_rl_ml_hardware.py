#!/usr/bin/env python3

import argparse
import logging
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from tf.transformations import quaternion_from_euler
from skimage.feature import peak_local_max

from hardware.armor_camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.target_plot import save_results, plot_results
from utils.dataset_processing.grasp import Grasp

import cv2
import cv2.aruco as aruco

import rospy
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs

logging.basicConfig(stream=sys.stdout,level=logging.INFO)

def deproject_pixel_to_point(depth, pixel, intrinsics):
    x = (pixel[0] - intrinsics.ppx) * depth / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) * depth / intrinsics.fy
    return np.array([x, y, depth])



def draw_rectangle(center, angle, length, width):

    length=40
    width=20
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = center[0] + length / 2 * yo
    x1 = center[1] - length / 2 * xo
    y2 = center[0] - length / 2 * yo
    x2 = center[1] + length / 2 * xo

    return (np.array(
        [
            [y1 - width / 2 * xo, x1 - width / 2 * yo],
            [y2 - width / 2 * xo, x2 - width / 2 * yo],
            [y1 + width / 2 * xo, x1 + width / 2 * yo],
            [y2 + width / 2 * xo, x2 + width / 2 * yo],
            
        ]
    ).astype(float))
        
def sample_points_along_line(p1, p2, num_points=10):
    """
    Generate `num_points` evenly spaced points between p1 and p2.
    """
    return np.linspace(p1, p2, num_points)

def is_target_red(grasp_point, color_image ):
    # Convert the color image to HSV color space
    color_image = np.copy(color_image)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    lower_red_1 = np.array([0, 120, 70])    # Lower range for red
    upper_red_1 = np.array([10, 255, 255])  # Upper range for red
    lower_red_2 = np.array([170, 120, 70])  # Second lower range for red
    upper_red_2 = np.array([180, 255, 255]) # Second upper range for red
    lower_blue = np.array([100, 150, 50])   # Lower range for blue
    upper_blue = np.array([140, 255, 255])  # Upper range for blue


    # Create masks for the two red ranges
    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 | mask2
    # red_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Check if the grasp_point (cx, cy) is within the red mask
    cx, cy = grasp_point
    if red_mask[cy, cx] > 0:
        # The point is in a red region
        return True
    else:
        # The point is not in a red region
        return False

def filter_grasps(grasps, img, depth_img, intrinsics, red_thresh=0, green_thresh=-1, blue_thresh=-1, num_depth_checks=10):
    """
    Filter out grasps that are not on the object or obstructed by depth.
    :param grasps: list of Grasps
    :param img: RGB Image # Shape: (3, H, W)
    :param depth_img: Depth Image # Shape: (H, W)
    :param red_thresh: Red threshold
    :param green_thresh: Green threshold
    :param blue_thresh: Blue threshold
    :param num_depth_checks: Number of depth checks from the center to the edge
    :return: list of Grasps
    """
    filtered_grasps = []
    for g in grasps:
        cy, cx = g.center
        grasp_point=(cx,cy)
        
        x=is_target_red(grasp_point, img)
        
        # Check color thresholds
        if x:
            
            center=[cx,cy]
            center_depth = depth_img[0,cy, cx]
            
            center_position = deproject_pixel_to_point(center_depth,center, intrinsics)
            
            # Draw rectangle points
            rect_points = draw_rectangle(center_position, g.angle, g.length, g.width)
            # print("length:",g.length)
            # print("width:",g.width)
            print("Red rect points:",rect_points)
            
            # Check depth constraint
            
            #print("Center:",center_depth)
            is_valid_grasp = True
            
            for x in range(0,len(rect_points)):
                # Sample points from center to each rectangle corner
                p1=rect_points[x]
                p2=rect_points[(x+2)%4]
                # print(p1)
                # print(p2)
                line_points = sample_points_along_line(p1, p2, num_depth_checks)
                for pt in line_points:
                    y,x = int(pt[0]), int(pt[1])
                    #print(depth_img[y,x])
                    #if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                    if depth_img[0,y, x] <= center_depth:
                        #print(y,x)
                        is_valid_grasp = False
                        print(is_valid_grasp)
                        break
                
            
            if is_valid_grasp:
                filtered_grasps.append(g)

    return filtered_grasps

def hardware_detect_grasps(q_img, ang_img,depth_img,rgb_img, intrinsics, width_img=None,no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    # Crop offset from 640x480 to 224x224
    crop_offset_x = 208  # (640 - 224) // 2
    crop_offset_y = 128  # (480 - 224) // 2
    grasps = []
    filtered_grasps = []
    for grasp_point_array in local_max:
        grasp_point_224 = tuple(grasp_point_array)
        #print(f'grasp point for 224x224: {grasp_point_224[1], grasp_point_224[0]}')  # x, y
        # Map grasp point back to 640x480 image
        grasp_point_640 = (grasp_point_224[1] + crop_offset_x, grasp_point_224[0] + crop_offset_y)  # Updating class variable
        #print(f'grasp for 640x480: {grasp_point_640}')
        
        grasp_angle = ang_img[grasp_point_224]
        g = Grasp(grasp_point_224, grasp_angle)
        
        if width_img is not None:
            g.length = width_img[grasp_point_224]
            g.width = g.length / 2
        grasps.append(g)
        
        filtered_grasps=filter_grasps(grasps, rgb_img, depth_img,intrinsics)
        print("filtered grasps:",filtered_grasps)
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

# ROS node
rospy.init_node('realsense_object_pose_publisher')
grasp_pub = rospy.Publisher('grasp_point', PoseStamped, queue_size=10)
logging.info('Ros node initialized')


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)
marker_size = 0.104 #100mm
transform_matrix = None


def rotation_matrix_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_matrix_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def aruco_transformation_matrix(T_x,T_y,T_z):
    
    theta_y = np.deg2rad(0) #clockwise rotation
    theta_x = np.deg2rad(0)

    R_y = rotation_matrix_y(theta_y)
    R_x = rotation_matrix_x(theta_x)
    R = R_y @ R_x #post multiply
    T_xyz = [T_x, T_y, T_z]

    T_ArUco_to_Bot = np.eye(4)
    T_ArUco_to_Bot[:3, :3] = R
    T_ArUco_to_Bot[:3, 3] = T_xyz
    
    return T_ArUco_to_Bot
    

def transform_object_to_bot(object_in_camera_frame, transform_matrix):
    object_in_camera_frame_homogeneous = np.append(object_in_camera_frame, [1])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    object_in_aruco_frame_homogeneous = np.dot(transform_matrix_inv, object_in_camera_frame_homogeneous)
    object_in_aruco_frame = object_in_aruco_frame_homogeneous[:3]
    return object_in_aruco_frame

def detect_aruco(color_image,camera_matrix,distortion_matrix):
    global marker_size
    global transform_matrix
    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY) #Cuz enable_stream reads it with rgb8 encoding
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        color_image = aruco.drawDetectedMarkers(color_image, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_matrix)
        for i in range(len(ids)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            rot_matrix, _ = cv2.Rodrigues(rvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = tvec.flatten()
    return transform_matrix

def pose_value_with_depth_compensation(grasp_point_640,depth_frame,depth_image,intrinsics):
    cx = grasp_point_640[0]
    cy = grasp_point_640[1]
    depth_value = depth_frame.get_distance(cx, cy)
    #depth_value = depth_image[cy,cx]
    x = (cx - intrinsics.ppx) / intrinsics.fx * depth_value
    y = (cy - intrinsics.ppy) / intrinsics.fy * depth_value
    return [x,y,depth_value]


class Graspable:
    def __init__(self, network_path, force_cpu=False):
        # Initialize necessary components and load the model
        self.force_cpu = force_cpu
        logging.info('Connecting to camera...')

        # Load the neural network model
        logging.info('Loading model...')
        if torch.cuda.is_available() and not force_cpu:
            self.net = torch.load(network_path)
        else:
            self.net = torch.load(network_path, map_location=torch.device('cpu'))

        logging.info('Model loaded successfully.')

        # Get the compute device
        self.device = get_device(force_cpu)

    def run_graspable(self, x, depth_image, denormalised_depth, rgb_img):
        # Run the grasp detection logic
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)

            # Post-process the network output
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            grasps = hardware_detect_grasps(rgb_img, denormalised_depth, q_img, ang_img, width_img=None, no_grasps=10)
            # # Visualize the results
            # fig = plt.figure(figsize=(10, 10))
            # plot_results(fig=fig,
            #              rgb_img=rgb_img,
            #              depth_img=np.squeeze(denormalised_depth),
            #              grasp_q_img=q_img,
            #              grasp_angle_img=ang_img,
            #              no_grasps=10,
            #              grasp_width_img=width_img,
            #              grasps=grasps)  
        return bool(len(grasps))


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera()
    cam.connect()
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

            rgb = image_bundle['rgb'] #640X480

            depth = image_bundle['aligned_depth'] 

            depth_unexpanded = image_bundle['unexpanded_depth']

            depth_frame = image_bundle['depth_frame']

            if transform_matrix is None:
                transform_matrix = detect_aruco(rgb,cam.camera_matrix,cam.distortion_matrix)
                print("Transform matrix has to be generated")          
            else:

                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth) #returns 224x224 rgb and depth images
                
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)

                    print("pred:",pred)

                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])


                    rgb_denorm=cam_data.get_rgb(rgb, False)
                    grasps = hardware_detect_grasps(q_img, ang_img,depth_img,rgb_denorm,cam.intrinsics, width_img=None, no_grasps=10)
                
                    plot_results(fig=fig,
                                 rgb_img=cam_data.get_rgb(rgb, False),
                                 depth_img=np.squeeze(cam_data.get_depth(depth)),
                                 grasp_q_img=q_img,
                                 grasp_angle_img=ang_img,
                                 no_grasps=args.n_grasps,
                                 grasp_width_img=width_img,
                                 grasps=grasps)

                    for grasp in grasps:
                        grasp_point_640 = (grasp.center[1]+208,grasp.center[0]+128)
                        grasp_angle = grasp.angle
                        object_in_camera_frame = pose_value_with_depth_compensation(grasp_point_640, depth_frame, depth_unexpanded, cam.intrinsics)
                        if object_in_camera_frame is not None and transform_matrix is not None:
                            object_in_aruco_frame = transform_object_to_bot(object_in_camera_frame, transform_matrix)
                            T_x,T_y,T_z = -0.28, -0.0, 0 #wrt aruco frame
                            transform_bot = np.eye(4)
                            tranform_bot = aruco_transformation_matrix(T_x,T_y,T_z)
                            object_in_bot_frame = transform_object_to_bot(object_in_aruco_frame,tranform_bot)
                            print(f"Object in ArUco marker frame: {object_in_aruco_frame}")
                            print(f"Object in Bot frame: {object_in_bot_frame}")
                            print(f'Grasp angle:',math.degrees(grasp_angle))#Angle in radians

                            quaternion = quaternion_from_euler(0,0,grasp_angle)
                            rx,ry,rz = object_in_bot_frame

                            # Publish to ROS topic
                            grasp_msg = PoseStamped()
                            grasp_msg.header.stamp = rospy.Time.now()
                            grasp_msg.pose.position.x = rx
                            grasp_msg.pose.position.y = ry
                            grasp_msg.pose.position.z = rz
                            print(f'pose stamped:::{rx,ry,rz}')
                            # Orientation will be published later
                            grasp_msg.pose.orientation.x = quaternion[0]
                            grasp_msg.pose.orientation.y = quaternion[1]
                            grasp_msg.pose.orientation.z = quaternion[2]
                            grasp_msg.pose.orientation.w = quaternion[3]

                            grasp_pub.publish(grasp_msg)
                            logging.info("published to object_pose topic")

    finally:
        save_results(
            rgb_img=cam_data.get_rgb(rgb, False),
            depth_img=np.squeeze(cam_data.get_depth(depth)),
            grasp_q_img=q_img,
            grasp_angle_img=ang_img,
            no_grasps=args.n_grasps,
            grasp_width_img=width_img)
