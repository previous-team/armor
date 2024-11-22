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

intrinsics = None


def deproject_pixel_to_point(depth, pixel):
    global intrinsics 
    print(intrinsics)
    x = (pixel[0] - intrinsics[0][2]) * depth / intrinsics[0][0]
    y = (pixel[1] - intrinsics[1][2]) * depth / intrinsics[1][1]
    return np.array([x, y, depth])

def project_point_to_pixel(depth, point):
    #x*fx/depth+ppx
    global intrinsics
    print(intrinsics)
    pixel_x=(point[1]*intrinsics[0][0]/depth)+intrinsics[0][2]
    pixel_y=(point[0]*intrinsics[1][1]/depth)+intrinsics[1][2]
    return pixel_x,pixel_y


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

def filter_grasps(grasps, img, depth_img, red_thresh=0, green_thresh=-1, blue_thresh=-1, num_depth_checks=10):
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
            
            angles = np.linspace(0, 2 * np.pi, 16) # 16 angles for the rectangle
            # Insert obtained angle from ML to the first index
            angles = np.insert(angles, 0, g.angle)
            
           # Check if it is graspable at any angle
            for angle in angles:
                center=[cx,cy]
                depth=depth_img[cy, cx]
                center_position = deproject_pixel_to_point(depth, center)
                # Draw rectangle points
                rect_points = draw_rectangle(center_position, g.angle, g.length, g.width)
                
                pixel_points=[]
                for pt in rect_points:
                    p_p=project_point_to_pixel(depth, pt)
                    pixel_points.append(p_p)
                
                # Check depth constraint
                center_depth = depth_img[cy, cx]
                is_valid_grasp = True
                
                for x in range(0,len(rect_points)):
                    # Sample points from center to each rectangle corner
                    p1=pixel_points[x]
                    p2=pixel_points[(x+2)%4]
                    line_points = sample_points_along_line(p1, p2, num_depth_checks)
                    for pt in line_points:
                        y,x = int(pt[0]), int(pt[1])
                        if (0 <= x < 224) and (0 <= y < 224) and depth_img[y, x] <= center_depth:
                            is_valid_grasp = False
                            break
                    
                
                if is_valid_grasp:
                    g.angle = angle  # Update the angle
                    filtered_grasps.append(g)
                    break

    return filtered_grasps

def hardware_detect_grasps(q_img, ang_img,depth_img,rgb_img, width_img=None,no_grasps=1):
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
        
        filtered_grasps=filter_grasps(grasps, rgb_img, depth_img)
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

def detect_aruco(color_image,camera_matrix,distortion_matrix,detector,marker_size):
    #global marker_size
    #global transform_matrix
    global intrinsics
    intrinsics = camera_matrix
    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY) #Cuz enable_stream reads it with rgb8 encoding
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        color_image = aruco.drawDetectedMarkers(color_image, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size,intrinsics, distortion_matrix)
        for i in range(len(ids)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            rot_matrix, _ = cv2.Rodrigues(rvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = tvec.flatten()
    return transform_matrix

def pose_value_with_depth_compensation(grasp_point_640,depth_frame,depth_image):
    global intrinsics
    cx = grasp_point_640[0]
    cy = grasp_point_640[1]
    depth_value = depth_frame.get_distance(cx, cy)
    #depth_value = depth_image[cy,cx]
    x = (cx - intrinsics[0][2]) / intrinsics[0][0] * depth_value
    y = (cy - intrinsics[1][2]) / intrinsics[1][1] * depth_value
    return [x,y,depth_value]


class Graspable:
    def __init__(self, network_path, force_cpu=False):
        # Initialize necessary components and load the model
        self.fig = plt.figure(figsize=(10, 10))
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
        
        self.grasp_pub = rospy.Publisher('grasp_point', PoseStamped, queue_size=10)

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(aruco_dict, aruco_params)
        self.marker_size = 0.104 #100mm
    

    def run_graspable(self, x, depth_image, denormalised_depth,rgb_img):
        # Run the grasp detection logic
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)

            # Post-process the network output
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            grasps = hardware_detect_grasps(q_img, ang_img,denormalised_depth, rgb_img, width_img=None, no_grasps=10)
            plot_results(fig=self.fig,
                rgb_img=rgb_img,
                depth_img=np.squeeze(depth_image),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=1,
                grasp_width_img=width_img,
                grasps=grasps)
            print(f"grasps{grasps}")
        return grasps
    
    def aruco_marker_detect(self, rgb, camera_matrix, distortion_matrix):
        global intrinsics 
        intrinsics = camera_matrix
        transform_matrix = self.detect_aruco(rgb,intrinsics, distortion_matrix)   
        return transform_matrix

    def detect_aruco(self, color_image, camera_matrix, distortion_matrix):
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY) # Enable stream reads it with rgb8 encoding
        corners, ids, _ = self.detector.detectMarkers(gray)
        transform_matrix = None  # Initialize in case no markers are detected

        if ids is not None and len(ids) > 0:
            color_image = aruco.drawDetectedMarkers(color_image, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, distortion_matrix)
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                rot_matrix, _ = cv2.Rodrigues(rvec)
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = tvec.flatten()
        
        return transform_matrix
        
    
    
    def pick(self,grasp,depth_frame,depth_unexpanded,transform_matrix):  
        grasp_point_224 = grasp.center
        grasp_angle = grasp.angle
        crop_offset_x = 208  # (640 - 224) // 2
        crop_offset_y = 128  # (480 - 224) // 2
        grasp_point_640 = (grasp_point_224[1] + crop_offset_x, grasp_point_224[0] + crop_offset_y)
        print(grasp_point_640,grasp_angle)
        
 
        object_in_camera_frame = pose_value_with_depth_compensation(grasp_point_640, depth_frame, depth_unexpanded)
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

            # # Publish to ROS topic
            # grasp_msg = PoseStamped()
            # grasp_msg.header.stamp = rospy.Time.now()
            # grasp_msg.pose.position.x = rx
            # grasp_msg.pose.position.y = ry
            # grasp_msg.pose.position.z = rz
            # print(f'pose stamped:::{rx,ry,rz}')
            
            # Orientation will be published later
            # grasp_msg.pose.orientation.x = quaternion[0]
            # grasp_msg.pose.orientation.y = quaternion[1]
            # grasp_msg.pose.orientation.z = quaternion[2]
            # grasp_msg.pose.orientation.w = quaternion[3]

            # self.grasp_pub.publish(grasp_msg)
        return rx,ry,rz,grasp_angle
        



