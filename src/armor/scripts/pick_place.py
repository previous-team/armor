#!/usr/bin/env python

# Imports
from niryo_robot_python_ros_wrapper import *
import rospy

# Initializing ROS node
rospy.init_node('niryo_robot_example_python_ros_wrapper')

# Connecting to the ROS Wrapper & calibrating if needed
niryo_robot = NiryoRosWrapper()
niryo_robot.calibrate_auto()

# Updating tool
niryo_robot.update_tool()

# Opening Gripper/Pushing Air
niryo_robot.release_with_tool()
# Going to pick pose
niryo_robot.move_pose(0.2, 0.1, 0.1, 0.0, 1.57, 0)

print(niryo_robot.get_pose(),type(niryo_robot.get_pose()))

# Picking
niryo_robot.grasp_with_tool()
# Moving to place pose
niryo_robot.move_pose(0.0, 0.2, 0.2, 0.0, 1.57, 0)
# Placing !
niryo_robot.release_with_tool()

# Home postion
niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)