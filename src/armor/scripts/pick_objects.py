#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R
from niryo_robot_python_ros_wrapper import NiryoRosWrapper

import rospy
from geometry_msgs.msg import PoseStamped


grasp_point_buffer = []

def camera_to_world(camera_coords, camera_pose):
    """
    Convert coordinates from camera frame to world frame.

    :param camera_coords: numpy array of shape (3,), the (x, y, z) coordinates in the camera frame
    :param camera_pose: dictionary with keys 'position' and 'orientation'
                        'position' is a numpy array of shape (3,)
                        'orientation' is a numpy array of shape (4,) for quaternion or (3,) for Euler angles
    :return: numpy array of shape (3,), the (x, y, z) coordinates in the world frame
    """
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
    camera_coords_homogeneous = np.append(camera_coords, 1)

    # Apply the transformation
    world_coords_homogeneous = np.dot(transformation_matrix, camera_coords_homogeneous)

    # Extract the world coordinates
    world_coords = world_coords_homogeneous[:3]

    return world_coords


def grasp_object_callback(msg):
    # Extract the position and orientation of the object
    # object_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    # # x is z; y is -x; z is -y
    object_position = np.array([msg.pose.position.z, -msg.pose.position.x, -msg.pose.position.y])
    object_orientation= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])



    # Create the camera pose
    camera_pose = {
        'position': np.array([0.3, 0, 0.55]),
        'orientation': np.array([0.0000001, 1.57, -3.141591])  # Euler angles (roll, pitch, yaw)
    }


    # Convert the object position from the camera frame to the world frame
    object_position_world = camera_to_world(object_position, camera_pose)

    # Convert orientation from camera frame to world frame (assuming it's only yaw)
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    
    r = R.from_quat(object_orientation)
    euler_angles = r.as_euler('xyz', degrees=False)

    #no translation as robot z axis and camera z axis same

    

    # Add position and yaw to the buffer
    if len(grasp_point_buffer) < 3:
        grasp_point_buffer.append((object_position_world, euler_angles[2])) 
    else:
        grasp_point_buffer.pop(0)
        grasp_point_buffer.append((object_position_world, euler_angles[2]))

    print("Object Position (World Frame):", object_position_world)
    print("Object Yaw (World Frame):", euler_angles[2])

def niryo_robot_pick_object(robot, object_position, grasp_angle):

    # Opening Gripper
    robot.release_with_tool()
    
    # Move to grasp pose (including yaw)
    robot.move_pose(object_position[0], object_position[1], 0.09, 0.0, 1.57, grasp_angle)

    # Picking
    robot.grasp_with_tool()

    rospy.sleep(1)

    # Home postion
    robot.move_joints(0, 0.5, -1.25, 0, 0, 0)

    # Moving to place pose
    robot.move_pose(0.0, 0.2, 0.2, 0.0, 1.57, 0)

    # Placing !
    robot.release_with_tool()

    # Home postion
    robot.move_joints(0, 0.5, -1.25, 0, 0, 0)


def main():

    # Initializing ROS node
    rospy.init_node('niryo_robot_pick_objects', anonymous=True)

    # Connecting to the ROS Wrapper & calibrating if needed
    niryo_robot = NiryoRosWrapper()
    niryo_robot.calibrate_auto()

    # Home position
    niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)

    # Subscribing to the grasp object topic
    rospy.Subscriber('/grasp_point', PoseStamped, grasp_object_callback)

    while not rospy.is_shutdown():
        if len(grasp_point_buffer) == 3:
            # Check if all the points are the same
            if all(np.allclose(grasp_point_buffer[0][0], point[0]) for point in grasp_point_buffer):
                print("Grasping object at:", grasp_point_buffer[0][0])
                print("Grasp angle:", grasp_point_buffer[0][1])
                
                # Pass position and yaw to the pick function
                niryo_robot_pick_object(niryo_robot, grasp_point_buffer[0][0], grasp_point_buffer[0][1])
                
                grasp_point_buffer.clear()
            rospy.sleep(1)


if __name__ == '__main__':
    main()
