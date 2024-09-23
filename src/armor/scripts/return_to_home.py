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

# # Moving to place pose
# niryo_robot.move_pose(0.046,-0.019273479107925352 ,0.012009209982891123+, 0.0, 1.57, 0)

# Home postion
niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
