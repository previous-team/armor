#!/usr/bin/env python

from niryo_robot_python_ros_wrapper import *
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

# Initialize ROS node
rospy.init_node('niryo_robot')

# Connect to the ROS Wrapper & calibrate if needed
niryo_robot = NiryoRosWrapper()
niryo_robot.calibrate_auto()
niryo_robot.update_tool()

# Opening Gripper/Pushing Air
niryo_robot.release_with_tool()


def grasp_callback(msg):
    object_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    object_orientation= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    r = R.from_quat(object_orientation)
    euler_angles = r.as_euler('xyz', degrees=False)
    x, y, z = object_position
    print(x,y,z)
    niryo_robot.move_pose(x, y,max(z+0.07,0.09), 0.0, 1.57, euler_angles[2])
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
rospy.Subscriber('/grasp_point', PoseStamped, grasp_callback)

# Keep the node running
rospy.spin()
