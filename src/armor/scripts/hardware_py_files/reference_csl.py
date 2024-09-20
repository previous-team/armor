#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		    Cosmo Logistic (CL) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script should be used to implement Task 1A of Cosmo Logistic (CL) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          2029
# Author List:		Archanaa A Chandaragi, Arjun Achar , Pradeep J S , Sanraj Lacchiramka
# Filename:		    task1a.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#                   Example:
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    area of detected aruco
        width       (float):    width of detected aruco
    '''

    ############ Function VARIABLES ############

    # You can remove these variables after reading the instructions. These are just for sample.

    area = None
    width = None

    ############ ADD YOUR CODE HERE ############

    # INSTRUCTIONS & HELP : 
    #	->  Recevice coordiantes from 'detectMarkers' using cv2.aruco library 
    #       and use these coordinates to calculate area and width of aruco detected.
    #	->  Extract values from input set of 4 (x,y) coordinates 
    #       and formulate width and height of aruco detected to return 'area' and 'width'.

    ############################################
    height = math.sqrt((coordinates[0][0] - coordinates[1][0])**2 + (coordinates[0][1] - coordinates[1][1])**2)
    width = math.sqrt((coordinates[1][0] - coordinates[2][0])**2 + (coordinates[1][1] - coordinates[2][1])**2)

    area = height * width

    return area, width


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    # Use this variable as a threshold value to detect aruco markers of certain size.
    # Ex: avoid markers/boxes placed far away from arm's reach position  
    aruco_area_threshold = 1500

    # The camera matrix is defined as per camera info loaded from the plugin used. 
    # You may get this from /camer_info topic when camera is spawned in gazebo.
    # Make sure you verify this matrix once if there are calibration issues.
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])

    # The distortion matrix is currently set to 0. 
    # We will be using it during Stage 2 hardware as Intel Realsense Camera provides these camera info.
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

    # We are using 150x150 aruco marker size
    size_of_aruco_m = 0.15

    # You can remove these variables after reading the instructions. These are just for sample.
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []
 
    ############ ADD YOUR CODE HERE ############

    # INSTRUCTIONS & HELP : 

    #	->  Convert input BGR image to GRAYSCALE for aruco detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #   ->  Use these aruco parameters-
    #       ->  Dictionary: 4x4_50 (4x4 only until 50 aruco IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters_create()

    #   ->  Detect aruco marker in the image and store 'corners' and 'ids'
    #       ->  HINT: Handle cases for empty markers detection. 
    corners,ids_list,empty_markers = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=params)

    #   ->  Draw detected marker on the image frame which will be shown later
    cv2.aruco.drawDetectedMarkers(image, corners, ids_list)

    #   ->  Loop over each marker ID detected in frame and calculate area using function defined above (calculate_rectangle_area(coordinates))
    if ids_list is not None:
        for i in range(len(ids_list)):
            area,width = calculate_rectangle_area(corners[i][0])
            #   ->  Remove tags which are far away from arm's reach positon based on some threshold defined
            if(area<aruco_area_threshold):
                continue
            else:
                #->  Calculate center points aruco list using math and distance from RGB camera using pose estimation of aruco marker
                #->  HINT: You may use numpy for center points and 'estimatePoseSingleMarkers' from cv2 aruco library for pose estimation

                cx = np.mean(corners[i][0][:,0])
                cy = np.mean(corners[i][0][:,1])
                center_aruco_list.append([cx,cy])
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)
                #print("center:",center_aruco_list[i])
                # distance_from_rgb = np.linalg.norm(tvec[0][0])
                distance_from_rgb = np.sqrt(tvec[0][0][0]**2 + tvec[0][0][1]**2 + tvec[0][0][2]**2)
                distance_from_rgb_list.append(distance_from_rgb)

                # Calculate the angle of the ArUco marker
                
                # angle_aruco = np.arctan2(np.linalg.norm([tvec[0][0][0],tvec[0][0][1]]), tvec[0][0][2])
                angle_aruco = rvec[0][0][2] # yaw
                angle_aruco_list.append(angle_aruco)
                '''
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                angles_rad = cv2.RQDecomp3x3(rotation_matrix)[0]
                angles_deg = list(np.degrees(angles_rad))
                angle_aruco_list.append(angles_deg)
                '''

                # Append marker width and ID
                width_aruco_list.append(width)
                ids.append(ids_list[i][0])

                #->  Draw frame axes from coordinates received using pose estimation
                #->  HINT: You may use 'cv2.drawFrameAxes'
                cv2.aruco.drawAxis(image, cam_mat, dist_mat, rvec, tvec, size_of_aruco_m)
    else:
        print("No ArUco marker detected")
    ############################################


    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids


##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
                        All classes have a function called __init__(), which is always executed when the class is being initiated.
                        The __init__() function is called automatically every time the class is being used to create a new object.
                        You can find more on this topic here -> https://www.w3schools.com/python/python_classes.asp
        '''

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        ############ Publishers #############
        
        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())


    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image

        Args:
            data (Image):    Input depth image frame received from aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT: You may use CvBridge to do the same

        ############################################
        
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
        


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image

        Args:
            data (Image):    Input coloured raw image frame received from image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT:   You may use CvBridge to do the same
        #               Check if you need any rotation or flipping image as input data maybe different than what you expect to be.
        #               You may use cv2 functions such as 'flip' and 'rotate' to do the same

        ############################################
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)


    def process_image(self):
        '''
        Description:    Timer function used to detect aruco markers and publish tf on estimated poses.

        Args:
        Returns:
        '''

        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
            

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #	->  Get aruco center, distance from rgb, angle, width and ids list from 'detect_aruco_center' defined above
        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(self.cv_image)
        #print(center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids)

        #   ->  Loop over detected box ids received to calculate position and orientation transform to publish TF 
        for i in range(len(ids)):
            #->  Use this equation to correct the input aruco angle received from cv2 aruco function 'estimatePoseSingleMarkers' here
            #       It's a correction formula- 
            #       angle_aruco = (0.788*angle_aruco) - ((angle_aruco**2)/3160)
            #print(angle_aruco_list[i])
            angle_aruco = (0.788*int(angle_aruco_list[i])) - ((int(angle_aruco_list[i])**2)/3160)
            #->  Then calculate quaternions from roll pitch yaw (where, roll and pitch are 0 while yaw is corrected aruco_angle)
            qx,qy,qz,qw = R.from_euler('xyz', [math.pi/2, 0, ((math.pi/2)-angle_aruco)]).as_quat()
            # qx,qy,qz,qw = R.from_euler('xyz', [90, 0, 90], degrees=True).as_quat()


            #->  Use center_aruco_list to get realsense depth and log them down. (divide by 1000 to convert mm to m)
            cX = center_aruco_list[i][0]
            cY = center_aruco_list[i][1]
            distance_from_rgb = distance_from_rgb_list[i]

            #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
            
            x = (distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX)
            y = (distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY)
            z = (distance_from_rgb)

            x,y,z = z,x,y

            #print(x,y,z)

            #       where, 
            #               cX, and cY from 'center_aruco_list'
            #               distance_from_rgb is depth of object calculated in previous step
            #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above
            center_coords = (int(cX),int(cY))
            #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.cirle' function
            cv2.circle(self.cv_image,center_coords,10,(255,0,0),-1)
            cv2.putText(self.cv_image,"center",center_coords,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            #   ->  Here, till now you receive coordinates from camera_link to aruco marker center position. 

            #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            #       so that we will collect it's position w.r.t base_link in next step.
            #       Use the following frame_id-
            #           frame_id = 'camera_link'
            #           child_frame_id = 'cam_<marker_id>'          Ex: cam_20, where 20 is aruco marker ID
            t.header.frame_id = 'camera_link'
            t.child_frame_id = 'cam_'+str(ids[i])

            # translation
            t.transform.translation.x = x
            t.transform.translation.y = y            
            t.transform.translation.z = z
            # rotation
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)

            #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
            #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 
            
            from_frame_rel = 'cam_'+str(ids[i])                                              
            to_frame_rel = 'base_link'                                                                   

            try:
                t1 = self.tf_buffer.lookup_transform( to_frame_rel, from_frame_rel, rclpy.time.Time())       
                self.get_logger().info(f'Successfully received data!')
            except tf2_ros.TransformException as e:
                self.get_logger().info("Could not transform "+from_frame_rel+" to "+to_frame_rel)
                continue
            
            
            #   ->  And now publish TF between object frame and base_link
            #       Use the following frame_id-
            #           frame_id = 'base_link'
            #           child_frame_id = 'obj_<marker_id>'          Ex: obj_20, where 20 is aruco marker ID

            #t2 = TransformStamped()
            #t2.header.stamp = self.get_clock().now().to_msg()
            #       so that we will collect it's position w.r.t base_link in next step.
            #       Use the following frame_id-
            #           frame_id = 'camera_link'
            #           child_frame_id = 'cam_<marker_id>'          Ex: cam_20, where 20 is aruco marker ID

            t2 = TransformStamped()
            t2.header.stamp = self.get_clock().now().to_msg()

            t2.header.frame_id = 'base_link'
            t2.child_frame_id = 'obj_'+str(ids[i])

            # # translation
            t2.transform.translation.x = t1.transform.translation.x
            t2.transform.translation.y = t1.transform.translation.y                
            t2.transform.translation.z = t1.transform.translation.z
            # # rotation
            t2.transform.rotation.x = qx
            t2.transform.rotation.y = qy
            t2.transform.rotation.z = qz
            t2.transform.rotation.w = qw

            self.br.sendTransform(t2)
            

            #listener for obj_id transform

            from_frame_rel = 'obj_'+str(ids[i])                                                                        # frame from which transfrom has been sent
            to_frame_rel = 'base_link'                                                                      # frame to which transfrom has been sent

            try:
                t3 = self.tf_buffer.lookup_transform( to_frame_rel, from_frame_rel, rclpy.time.Time())   # look up for the transformation between 'obj_1' and 'base_link' frames
                self.get_logger().info(f'Successfully received data!')
            except tf2_ros.TransformException as e:
                self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {e}')
                return

            # Logging transform data...
            print("For box__"+str(ids[i]))
            self.get_logger().info(f'Translation X:  {t3.transform.translation.x}')
            self.get_logger().info(f'Translation Y:  {t3.transform.translation.y}')
            self.get_logger().info(f'Translation Z:  {t3.transform.translation.z}')
            self.get_logger().info(f'Rotation X:  {t3.transform.rotation.x}')                                # NOTE: rotations are in quaternions
            self.get_logger().info(f'Rotation Y:  {t3.transform.rotation.y}')
            self.get_logger().info(f'Rotation Z:  {t3.transform.rotation.z}')
            self.get_logger().info(f'Rotation W:  {t3.transform.rotation.w}')
            

            #   ->  At last show cv2 image window having detected markers drawn and center points located using 'cv2.imshow' function.
            #       Refer MD book on portal for sample image -> https://portal.e-yantra.org/
            cv2.imshow("Color image",self.cv_image)
            cv2.waitKey(1)

        #   ->  NOTE:   The Z axis of TF should be pointing inside the box (Purpose of this will be known in task 1B)
        #               Also, auto eval script will be judging angular difference aswell. So, make sure that Z axis is inside the box (Refer sample images on Portal - MD book)

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/
    '''

    main()
