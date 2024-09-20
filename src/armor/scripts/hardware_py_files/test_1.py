#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

import rospy
from std_msgs.msg import Float32MultiArray


# ROS node
rospy.init_node('realsense_object_pose_publisher')
pose_pub = rospy.Publisher('object_pose', Float32MultiArray, queue_size=10)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color)
intrinsics = video_stream.as_video_stream_profile().get_intrinsics()
print(intrinsics)

camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
distortion_matrix = np.array(intrinsics.coeffs)


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

def pose_value_with_depth_compensation(color_image, depth_frame, intrinsics,resolution):
    object_in_camera_frame = None
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 3)
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)

            depth_value = depth_frame.get_distance(cx, cy)
            x = (cx - intrinsics.ppx) / intrinsics.fx * depth_value
            y = (cy - intrinsics.ppy) / intrinsics.fy * depth_value
            object_in_camera_frame = [x, y, depth_value]

            '''if considering resolution use this'''
            # # Normalizing pixel coordinates using resolution
            # width, height = resolution
            # x_norm = (cx - intrinsics.ppx) / width
            # y_norm = (cy - intrinsics.ppy) / height

            # # Scaling normalized coordinates using depth to account for parallax
            # x = x_norm * depth_value * width / intrinsics.fx
            # y = y_norm * depth_value * height / intrinsics.fy
            # object_in_camera_frame = [x, y, depth_value]

        return object_in_camera_frame

def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), (255, 0, 0), 3)

    return None

try:
    align = rs.align(rs.stream.color)
    b=1
    while b:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        b=0 
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY) 
    corners , ids , _ = detector.detectMarkers(gray)
    transform_matrix = np.eye(4)
    transform_bot = np.eye(4)

    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    transform_matrix = None
    if ids is not None and len(ids) > 0:
        color_image = aruco.drawDetectedMarkers(color_image, corners, ids)
        marker_size = 0.1  #100mm
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_matrix)

        for i in range(len(ids)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            rot_matrix, _ = cv2.Rodrigues(rvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = tvec.flatten()

            # print(f"Updated Homogeneous Transformation Matrix:\n{transform_matrix}")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        #Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        #detecting object's position in the camera frame
        object_in_camera_frame = pose_value_with_depth_compensation(color_image,depth_frame,intrinsics,(1280,720))
        print(f"Object in camera frame: {object_in_camera_frame}")


        if object_in_camera_frame is not None and transform_matrix is not None:
            object_in_aruco_frame = transform_object_to_bot(object_in_camera_frame, transform_matrix)
            T_x,T_y,T_z = -0.2725,0,0 #wrt aruco frame
            tranform_bot = aruco_transformation_matrix(T_x,T_y,T_z)
            object_in_bot_frame = transform_object_to_bot(object_in_aruco_frame,tranform_bot)
            print(f"Object in ArUco marker frame: {object_in_aruco_frame}")
            print(f"Object in Bot frame: {object_in_bot_frame}")

            # Publish to ROS topic
            pose_msg = Float32MultiArray(data=object_in_bot_frame)
            pose_pub.publish(pose_msg)

        draw_axis(color_image, camera_matrix, None, rvec, tvec, 0.1)
        cv2.imshow("Aruco with masking for red objects", color_image)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
