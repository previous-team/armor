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
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import Grasp
from utils.dataset_processing.grasp import hardware_detect_grasps

import cv2
import cv2.aruco as aruco

import rospy
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs

logging.basicConfig(stream=sys.stdout,level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='/home/archanaa/armor/capstone_armor/src/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
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
marker_size = 0.1 #100mm
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
                print("Transform matrix generated")          
            else:

                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth) #returns 224x224 rgb and depth images
                
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)

                    print("pred:",pred)

                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                    # grasps = hardware_detect_grasps(q_img, ang_img, width_img, no_grasps=1,img=rgb_img)
                
                    grasps,grasp_param= plot_results(fig=fig,
                                 rgb_img=cam_data.get_rgb(rgb, False),
                                 depth_img=np.squeeze(cam_data.get_depth(depth)),
                                 grasp_q_img=q_img,
                                 grasp_angle_img=ang_img,
                                 no_grasps=args.n_grasps,
                                 grasp_width_img=width_img)
                
                                    
                    for grasp in grasps:
                        grasp_point_224 = grasp.center
                        grasp_angle = grasp.angle
                        crop_offset_x = 208  # (640 - 224) // 2
                        crop_offset_y = 128  # (480 - 224) // 2
                        grasp_point_640 = (grasp_point_224[1] + crop_offset_x, grasp_point_224[0] + crop_offset_y)
                        print(f'Grasp point for 640x480: {grasp_point_640}')
                        object_in_camera_frame = pose_value_with_depth_compensation(grasp_point_640, depth_frame, depth_unexpanded, cam.intrinsics)
                        if object_in_camera_frame is not None and transform_matrix is not None:
                            object_in_aruco_frame = transform_object_to_bot(object_in_camera_frame, transform_matrix)
                            T_x,T_y,T_z = -0.278,-0.014,0 #wrt aruco frame
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
