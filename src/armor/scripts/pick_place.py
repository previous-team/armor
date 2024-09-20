#!/usr/bin/env python

from niryo_robot_python_ros_wrapper import *
import rospy
from std_msgs.msg import Float32MultiArray

# Initialize ROS node
rospy.init_node('niryo_robot_example_python_ros_wrapper')

# Connect to the ROS Wrapper & calibrate if needed
niryo_robot = NiryoRosWrapper()
niryo_robot.calibrate_auto()
niryo_robot.update_tool()

# Opening Gripper/Pushing Air
niryo_robot.release_with_tool()


def pose_callback(msg):
    x, y, z = msg.data
    print(x,y,z)
    niryo_robot.move_pose(x, y,z+0.07, 0.0, 1.57, 0)
    rospy.sleep(2)
    # Picking
    niryo_robot.grasp_with_tool()

    # Move back to home position
    niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)   

    # Moving to place pose
    niryo_robot.move_pose(0.0, 0.2, 0.2, 0.0, 1.57, 0)
    # Placing !
    niryo_robot.release_with_tool()

    # Home postion
    niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)

    rospy.signal_shutdown("Pose reached and home position set.")

# Subscribe to the topic with the object pose
rospy.Subscriber('/grasp_point', Float32MultiArray, pose_callback)

# Keep the node running
rospy.spin()
