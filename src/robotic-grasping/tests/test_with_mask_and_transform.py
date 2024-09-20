#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# ArUco marker dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)  #Instantiating detector

# Retrieve intrinsic parameters using SDK
profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color)  # Get the color stream profile
intrinsics = video_stream.as_video_stream_profile().get_intrinsics()

# Convert the intrinsics to a camera matrix
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])

# Function to transform the object position from camera frame to ArUco marker frame

def transform_object_to_aruco(object_in_camera_frame, transform_matrix):
    # Convert the object position to homogeneous coordinates
    object_in_camera_frame_homogeneous = np.append(object_in_camera_frame, [1])

    # Compute the inverse of the transformation matrix (to get T_aruco_to_camera)
    transform_matrix_inv = np.linalg.inv(transform_matrix)

    # Transform the object position to the ArUco marker frame
    object_in_aruco_frame_homogeneous = np.dot(transform_matrix_inv, object_in_camera_frame_homogeneous)

    # Convert back to 3D coordinates
    object_in_aruco_frame = object_in_aruco_frame_homogeneous[:3]  # Ignore the homogeneous coordinate (last value)

    return object_in_aruco_frame

# Function to detect red object and get its position in the camera frame
def pose_value(color_image, depth_frame, intrinsics):
    object_in_camera_frame = None  # Initialize to None if no object is detected

    # Convert color image to HSV for color detection
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for red color
    lower_red_1 = np.array([0, 120, 70])   # Lower range for red
    upper_red_1 = np.array([10, 255, 255]) # Upper range for red
    lower_red_2 = np.array([170, 120, 70]) # Lower range for red
    upper_red_2 = np.array([180, 255, 255]) # Upper range for red

    # Create a binary mask for the red color
    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 | mask2

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour 
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # X-coordinate in image plane
            cy = int(M['m01'] / M['m00'])  # Y-coordinate in image plane

            # Draw the contour and centroid on the color image
            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 3)
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)

            # Get the depth value at the centroid (Z-coordinate)
            depth_value = depth_frame.get_distance(cx, cy)

            # Convert pixel coordinates to real-world coordinates using depth
            object_in_camera_frame = rs.rs2_deproject_pixel_to_point(
                intrinsics, [cx, cy], depth_value
            )

    return object_in_camera_frame

# Function to draw custom axis
def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the axes (red, green, blue for X, Y, Z respectively)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis in red
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis in green
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis in blue
    return None

try:
    # Wait for frames
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Detect the object's position in the camera frame
        object_in_camera_frame = pose_value(color_image, depth_frame, intrinsics)
        print(f"Object in camera frame: {object_in_camera_frame}")

        # Detect ArUco markers in the image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)  # Updated function call

        if ids is not None and len(ids) > 0:
            # Draw markers
            color_image = aruco.drawDetectedMarkers(color_image, corners, ids)

            # Estimate pose of each marker
            marker_size = 0.05  # Define marker size (in meters)

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, None)

            # For each detected marker, calculate the transformation matrix
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Draw axis for the marker
                draw_axis(color_image, camera_matrix, None, rvec, tvec, 0.1)

                # Convert rotation vector to rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rvec)

                # Create homogeneous transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = tvec

                # print(f"Marker ID: {ids[i]}")
                # print(f"Homogeneous Transformation Matrix:\n{transform_matrix}")

                if object_in_camera_frame is not None:
                    # Calculate object position in the ArUco marker frame
                    object_in_aruco_frame = transform_object_to_aruco(object_in_camera_frame, transform_matrix)
                    print(f"Object Position in ArUco Marker Frame: {object_in_aruco_frame}")

        # Show the color image with markers and axes
        cv2.imshow('ArUco Marker Detection', color_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
