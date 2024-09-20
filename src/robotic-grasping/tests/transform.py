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
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Corrected for newer version
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)  # Instantiate detector

# Retrieve intrinsic parameters using SDK
profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color)  # Get the color stream profile
intrinsics = video_stream.as_video_stream_profile().get_intrinsics()

# Convert the intrinsics to a camera matrix
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
# Function to draw custom axis
def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the axes (red, green, blue for X, Y, Z respectively)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis in red
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis in green
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis in blue
    return img
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

        # Convert the color image to HSV color space
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the HSV range for red color
        lower_red_1 = np.array([0, 120, 70])   # Lower range for red
        upper_red_1 = np.array([10, 255, 255]) # Upper range for red
        lower_red_2 = np.array([170, 120, 70]) # Lower range for red
        upper_red_2 = np.array([180, 255, 255]) # Upper range for red

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
                # draw_axis(color_image, camera_matrix, None, rvec, tvec, 0.1)

                # Convert rotation vector to rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rvec)

                # Create homogeneous transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = tvec

                print(f"Marker ID: {ids[i]}")
                print(f"Homogeneous Transformation Matrix:\n{transform_matrix}")

        # Show the color image with markers and axes
        cv2.imshow('ArUco Marker Detection', color_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
