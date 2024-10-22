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
from utils.visualisation.target_plot import plot_results
from utils.dataset_processing.grasp import Grasp

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as R
from skimage.feature import peak_local_max
from utils.dataset_processing import image

logging.basicConfig(level=logging.INFO)


def deproject_pixel_to_point(depth, pixel, ppx, ppy, fx, fy):
    x = (pixel[0] - ppx) * depth / fx
    y = (pixel[1] - ppy) * depth / fy
    return np.array([x, y, depth])




def camera_to_world(camera_coords, camera_pose):
    """
    Convert coordinates from camera frame to world frame.

    :param camera_coords: numpy array of shape (3,), the (x, y, z) coordinates in the camera frame
    :param camera_pose: dictionary with keys 'position' and 'orientation'
                        'position' is a numpy array of shape (3,)
                        'orientation' is a numpy array of shape (4,) for quaternion or (3,) for Euler angles
    :return: numpy array of shape (3,), the (x, y, z) coordinates in the world frame
    """
    # Convert units to meters
    camera_coords = camera_coords / 1000

    # Apply the specific transformation: x becomes z, y becomes -x, z becomes -y
    rotation_matrix_specific = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    camera_coords_transformed = np.dot(rotation_matrix_specific, camera_coords)

    # Extract position and orientation
    position = camera_pose['position']
    orientation = camera_pose['orientation']
    
    # Check if orientation is given as quaternion or Euler angles
    if len(orientation) == 4:
        # Quaternion to rotation matrix
        rotation_matrix = R.from_quat(orientation).as_matrix()
    elif len(orientation) == 3:
        # Euler angles to rotation matrix
        rotation_matrix = R.from_euler('xyz', orientation).as_matrix()
    else:
        raise ValueError("Orientation must be a quaternion (4,) or Euler angles (3,)")

    # Create the homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    # Convert camera coordinates to homogeneous coordinates
    camera_coords_homogeneous = np.append(camera_coords_transformed, 1)

    # Apply the transformation
    world_coords_homogeneous = np.dot(transformation_matrix, camera_coords_homogeneous)

    # Extract the world coordinates
    world_coords = world_coords_homogeneous[:3]

    return world_coords

def project_point_to_pixel(depth, point, ppx, ppy, fx, fy):
    #x*fx/depth+ppx
    pixel_x=(point[1]*fx/depth)+ppx
    pixel_y=(point[0]*fy/depth)+ppy 
    return pixel_x,pixel_y

def draw_rectangle(center, angle, length, width):

    length=45
    width=23
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

def filter_grasps(grasps, img, depth_img, fx, fy, ppx, ppy, red_thresh=0, green_thresh=-1, blue_thresh=-1, num_depth_checks=10):
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
        # Check color thresholds
        if (img[0, cy, cx] > red_thresh and 
            img[1, cy, cx] > green_thresh and 
            img[2, cy, cx] > blue_thresh):
            
            
            center=[cx,cy]
            depth=depth_img[cy, cx]
            center_position = deproject_pixel_to_point(depth, center, ppx, ppy, fx, fy)
            # Draw rectangle points
            rect_points = draw_rectangle(center_position, g.angle, g.length, g.width)
            #print("Rectanngle points", rect_points)
            
            pixel_points=[]
            for pt in rect_points:
                p_p=project_point_to_pixel(depth, pt, ppx, ppy, fx, fy)
                pixel_points.append(p_p)
            print("Pixel_points:",pixel_points)
                
            
            #print("length:",g.length)
            #print("width:",g.width)
            #print(rect_points)
            
            # Check depth constraint
            center_depth = depth_img[cy, cx]
            #print("Center:",center_depth)
            is_valid_grasp = True
            
            for x in range(0,len(rect_points)):
                # Sample points from center to each rectangle corner
                p1=pixel_points[x]
                p2=pixel_points[(x+2)%4]
                print("p1:",p1)
                print("p2:",p2)
                line_points = sample_points_along_line(p1, p2, num_depth_checks)
                for pt in line_points:
                    y,x = int(pt[0]), int(pt[1])
                    #print(depth_img[y,x])
                    #if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                    if depth_img[y, x] <= center_depth:
                        #print("not graspable at pts:",(x,y))
                        is_valid_grasp = False
                        break
                
            
            if is_valid_grasp:
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
        
    fx = 462.1379699707031
    fy = 462.1379699707031
    ppx = 111
    ppy = 111
    

    filtered_grasps = filter_grasps(grasps, rgb_img , depth, fx, fy, ppx, ppy)

    for g in filtered_grasps:
        print("Grasp at: ", g.center, g.angle)

        cy,cx = g.center  # Invert to reverse the transpose operation that is given to the network

        g.angle = g.angle

        print("Grasp angle: ", math.degrees(g.angle))
        
        quaternion = quaternion_from_euler(0, 0, g.angle) #rotation about the z-axis

        z = depth[cy, cx]



        object_position = deproject_pixel_to_point(z, (cx, cy), ppx, ppy, fx, fy)

        # Create the camera pose
        camera_pose = {
            'position': np.array([0.3, 0, 0.55]),
            'orientation': np.array([0.0000001, 1.57, -3.141591])  # Euler angles (roll, pitch, yaw)
        }

        # Convert the object position from the camera frame to the world frame
        x, y, z = camera_to_world(object_position, camera_pose)

        print("Grasp at: ", x, y, z)

        # Publish the grasp point
        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = rospy.Time.now()
        grasp_msg.pose.position.x = x
        grasp_msg.pose.position.y = y
        grasp_msg.pose.position.z = z
        
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

    # Initialize ROS node
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
                             grasp_width_img=width_img,
                             grasps=grasps)
                
    finally:
        pass
