#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R
from niryo_robot_python_ros_wrapper import NiryoRosWrapper

import rospy
from geometry_msgs.msg import PoseStamped


grasp_point_buffer = []


def grasp_object_callback(msg):
    # Extract the position and orientation of the object
    object_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    object_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    # Convert orientation from camera frame to world frame (assuming it's only yaw)
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    r = R.from_quat(object_orientation)
    euler_angles = r.as_euler('xyz', degrees=False)

    # Add position and yaw to the buffer
    if len(grasp_point_buffer) < 5:
        grasp_point_buffer.append((object_position, euler_angles[2])) 
    else:
        grasp_point_buffer.pop(0)
        grasp_point_buffer.append((object_position, euler_angles[2]))

    print("Object Position (World Frame):", object_position)
    print("Object Yaw (World Frame):", euler_angles[2])

def niryo_robot_pick_object(robot, object_position, grasp_angle):

    # Opening Gripper
    robot.release_with_tool()
    
    # Move to grasp pose (including yaw)
    robot.move_pose(object_position[0], object_position[1], max(object_position[2] + 0.07, 0.09), 0.0, 1.57, grasp_angle)

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
        if len(grasp_point_buffer) == 5:
            # Check if all the points are the same
            if all(np.allclose(grasp_point_buffer[0][0], point[0], atol=0.01) for point in grasp_point_buffer):
                print("Grasping object at:", grasp_point_buffer[0][0])
                print("Grasp angle:", grasp_point_buffer[0][1])
                
                # Pass position and yaw to the pick function
                niryo_robot_pick_object(niryo_robot, grasp_point_buffer[0][0], grasp_point_buffer[0][1])
                
                grasp_point_buffer.clear()
            rospy.sleep(1)


if __name__ == '__main__':
    main()
