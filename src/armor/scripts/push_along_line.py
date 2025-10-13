#!/usr/bin/env python

# Imports
import math

from niryo_robot_python_ros_wrapper import *
import rospy

# Initializing ROS node
rospy.init_node('niryo_robot_push_along_line')

# Connecting to the ROS Wrapper & calibrating if needed
niryo_robot = NiryoRosWrapper()
niryo_robot.calibrate_auto()

# Updating tool
niryo_robot.update_tool()

# Function to push along line
def push_along_line(x, y, z, theta, length, debug=False):
    '''
    Pushes the robot along a line
    x: x coordinate of the starting position
    y: y coordinate of the starting position
    z: z coordinate of the starting position
    theta: angle of the line
    length: length of the line
    debug: boolean to print debug statements
    '''
    # Go to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
    if debug and res and res[0] == 1:
        print("Moved to home position")
    elif debug:
        print("Failed to move to home position")

    # Close the gripper
    res = niryo_robot.grasp_with_tool()
    while not res:
        res = niryo_robot.grasp_with_tool()
    if debug and res and res[0] == 1:
        print("Closed the gripper")
    elif debug:
        print("Failed to close the gripper")

    # Move to the starting position
    res = niryo_robot.move_pose(x, y, max(z + 0.07, 0.1), 0.0, 1.57, 0)
    if debug and res and res[0] == 1:
        print("Moved to the starting position: ", x, y, z)
    elif debug:
        print("Failed to move to the starting position")

    final_x, final_y = x + length * math.cos(theta), y + length * math.sin(theta)
    res = niryo_robot.move_pose(final_x, final_y, max(z + 0.07, 0.1), 0.0, 1.57, 0)
    if debug and res and res[0] == 1:
        print("Moved to the final position: ", final_x, final_y, z)
    elif debug:
        print("Failed to move to the final position")

    # Go to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
    if debug and res and res[0] == 1:
        print("Moved to home position")
    elif debug:
        print("Failed to move to home position")


if __name__ == '__main__':
    # Push along line
    push_along_line(0.2, 0.0, 0.0, math.radians(90), 0.2, debug=True)