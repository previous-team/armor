#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import math

#Initializing RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30) #640x480 pixels (width x height)
#depending on grasping model requirements should be edited(:todo)
config.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30) #30 frames per second and 16bit depth which means each pixel value represents the distance to the camera in millimeters
#color has 8 bits per channel


#Begin streaming
pipeline.start(config)
#Assigning aruco marker dicyionary parameter and the detector object
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#4x4 bits in size, with 50 unique markers in total
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict,aruco_params)
profile = pipeline.get_active_profile() #returns the current active profile, which contains information about the streams currently being processed
video_stream = profile.get_stream(rs.stream.color) #retrievig color stream
intrinsics = video_stream.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([[intrinsics.fx,0,intrinsics.ppx],
                          [0,intrinsics.fy,intrinsics.ppy],
                          [0,0,1]])
distortion_matrix = np.array(intrinsics.coeffs)
print(f'cam_matrix{camera_matrix}')

sizeCamX = 1280
sizeCamY = 720
centerCamX = intrinsics.ppx
centerCamY = intrinsics.ppy
focalX = intrinsics.fx
focalY = intrinsics.fy

def transform_object_to_bot(object_in_camera_frame,transform_matrix):
    #(:todo) Convesion of aruco to robot frame
    #currently it is only wrt aruco

    #converting the object position to homogeneous coordinates
    #Homogeneous coordinates add an additional component (usually 1) to the 3D point, resulting in [x, y, z, 1]. This is necessary for matrix transformations involving translations.
    object_in_camera_frame_homogeneous = np.append(object_in_camera_frame,[1])

    #Computing the inverse of the transformation matrix (to get T_aruco_to_camera)
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    #The inverse matrix will convert points from the camera frame to the ArUco marker frame.

    #Transforming xthe object position to aruco marker frame
    object_in_aruco_frame_homogeneous = np.dot(transform_matrix_inv,object_in_camera_frame_homogeneous)

    #converting back to 3D coordinates
    object_in_aruco_frame = object_in_aruco_frame_homogeneous[:3]

    return object_in_aruco_frame

"""
function name: pose_value
Purpose: Simple function to apply mask for red object and get centroid
arguments: color_image,depth_frame,intrinsics of camera
returns: object_in_camera_frame[x,y,z]
"""
# (:todo) Placeholder for grasping model input

def pose_value(color_image,depth_frame,intrinsics):
    object_in_camera_frame = None #Just in case no object is detected

    #Converting color image to HSV for color detection
    hsv_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
    #more effective for color detection since the hue represents the color, saturation represents the intensity of the color, and value represents the brightness. This separation makes it easier to detect specific colors (like red) across varying lighting conditions

    #hsv range for red color
    lower_red_1 = np.array([0,120,70]) #lower range for red near 0 degree          #     The red color appears in two separate regions of the hue spectrum in HSV, one at low hue values (around 0°) and another at high hue values (close to 180°).
    upper_red_1 = np.array([10,255,255]) #Upper range of red at 0 degree            # The ranges are split into two:
    lower_red_2 = np.array([170,120,70]) #lower range of red at 180 degree          # lower_red_1 to upper_red_1: Captures red hues near 0°.
    upper_red_2 = np.array([180,255,255]) #Upper range of red at 180 degree         # lower_red_2 to upper_red_2: Captures red hues near 180°.

    #Binary mask for red color
    mask1 = cv2.inRange(hsv_image,lower_red_1,upper_red_1)
    mask2 = cv2.inRange(hsv_image,lower_red_2,upper_red_2)
    red_mask = mask1 | mask2  #The logical OR (|) operation combines both masks to detect red pixels across both hue ranges.
    #Pixels that correspond to the color red in the original image are white (255).
    #All other pixels are black (0).

    #Find contours in the mask
    contours,_ = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.RETR_TREE retrieves all the contours and reconstructs a full hierarchy of nested contours (i.e., parent-child relationships between contours). This is useful when you want to analyze the relationship between different contours (e.g., one contour inside another).
    #cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. This reduces the number of points representing the contour, making it more efficient in terms of memory and processing time.

    if contours:
        #Finding the largest contour
        largest_contour = max(contours,key = cv2.contourArea)
        #Taking largest contour based on area size

        #Getting centroid of largest contour
        M = cv2.moments(largest_contour)
        #This function calculates the moments of a contour, which are certain weighted averages (scalars) of the contour's pixels. Moments help to compute various properties such as the area, centroid, and orientation of the contour.
        if M['m00'] != 0:
            # M['m00'] :The spatial moment representing the area of the contour.
            cx = int(M['m10'] / M['m00']) #X coordinate in image plane
            cy = int(M['m01']/M['m00']) # Y coordiante in image plane

            #Drawing the contour and centroid on the color image
            cv2.drawContours(color_image,[largest_contour],-1,(0,255,0),3)
            cv2.circle(color_image,(cx,cy),5,(255,0,0),-1)

            #Getting depth value at the centroid
            depth_value = depth_frame.get_distance(cx,cy)

            #Converting the pixel coordinates to real world coordinates
            object_in_camera_frame = rs.rs2_deproject_pixel_to_point(intrinsics,[cx,cy],depth_value)
            #rs.rs2_deproject_pixel_to_point(): A function provided by the RealSense SDK to convert 2D pixel coordinates to 3D coordinates based on depth.

    return object_in_camera_frame

"""
function_name: draw_axis
purpose: to draw axis
arguments: image,cam_matrix,distortion coefficients,rvec ,tvec ,The length of the axes
returns None
"""

def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):

    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the axes (Red - X, Green - Y, Blue - Z)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis in red
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis in green
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis in blue

    return None

try:
    b=1
    while b:
        #Fetch frame set of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        b=0 #for breaking once frame is read

    #Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Detecting the ArUco markers in the image
    gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY) #for noise reduction and efficiency
    corners , ids , _ = detector.detectMarkers(gray)
    # transform_matrix = np.eye(4)
    # transform_bot = np.eye(4)

    if ids is not None and len(ids) > 0:
        #Draw markers
        color_image = aruco.drawDetectedMarkers(color_image,corners,ids)
        #Estimating the pose of each marker
        marker_size = 0.05 # 50mm
        rvecs,tvecs,_ = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,distortion_matrix)

        # For each aruco marker detected transformation matrix is calculated but any way only one aruco will be there on the bot
        #(:todo)damage control for when the aruco is hidden by arm or place in a way where the cam cant miss it
        for i in range(len(ids)):
            cx = np.mean(corners[i][0][:,0])
            cy = np.mean(corners[i][0][:,1])
            
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            print(f'rvec{rvec}')
            
            #Drawing axis for the marker
            #here assuming no distortion cuz extra work
            draw_axis(color_image,camera_matrix,None,rvec,tvec,0.1)

            #Converting rotation vector to rotation matrx
            distance_from_rgb = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
            rot_matrix,_ = cv2.Rodrigues(rvec)
            x = (distance_from_rgb * (sizeCamX - cx - centerCamX) / focalX)
            y = (distance_from_rgb * (sizeCamY - cy - centerCamY) / focalY)
            z = (distance_from_rgb)
            print(f'rot mat{rot_matrix}')
            
            # angle_aruco = rvec[2]
            #Creating homogeneous transform matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3,:3] = rot_matrix
            transform_matrix[:3 , 3]=tvec.flatten() #flatten() is used to convert it to a 1D array with shape (3,)
            # print(f"Marker ID: {ids[i]}")
            print(f"Homogeneous Transformation Matrix to camera:\n{transform_matrix}")
            # translation = [0,-0.162,0.06]
            # Rx = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]])
            # Rz = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

            # # Multiply Rx and Rz
            # R_total = Rx @ Rz
            # transform_bot[:3,:3]=R_total
            # print(f'1{transform_bot}')
            # transform_bot[:3 , 3]=translation
            # print(f'tran2{transform_bot}')
            # cv2.imshow("Aruco",color_image)

    while True:
        #Fetch frame set of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        #Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        #detecting object's position in the camera frame
        object_in_camera_frame = pose_value(color_image,depth_frame,intrinsics)
        print(f"Object in camera frame: {object_in_camera_frame}")

        if object_in_camera_frame is not None:
            #Currentlt still only wrt to aruco
            object_in_aruco_frame = transform_object_to_bot(object_in_camera_frame,transform_matrix)
            # object_in_bot_frame = transform_object_to_bot(object_in_aruco_frame,transform_bot)
            print(f"Object in bot frame:{object_in_aruco_frame}")
        cv2.imshow("Aruco with masking for red objects",color_image)

        #Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 

