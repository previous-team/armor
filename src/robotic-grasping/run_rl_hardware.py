import rospy
import argparse
import numpy as np
import math
import cv2
import logging
from niryo_robot_python_ros_wrapper import *
from niryo_robot_utils import NiryoRosWrapperException
from stable_baselines3 import SAC
import torch.utils.data

from hardware.armor_camera import RealSenseCamera
from hardware.device import get_device
from utils.data.camera_data import CameraData
from run_rl_ml_hardware import Graspable
from run_rl_sac_with_clutter import NiryoRobotEnv
from stable_baselines3.common.vec_env import DummyVecEnv



def parse_args():
    '''
    Parse the arguments for the script
    '''
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='src/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
                        help='Path to saved network to evaluate')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def denormalize_action(action, action_min, action_max, normalized_min=0):
    '''
    Denormalizes the action values to the real-world values.
    action: the normalized action value
    action_min: minimum value of the action
    action_max: maximum value of the action
    '''
    if normalized_min == -1:
        return action_min + 0.5 * (action_max - action_min) * (1 + action)
    else:
        return action_min + (action_max - action_min) * action

def go_to_home_position(debug=False):
    '''
    Moves the robot to the home position
    '''
    try:
        res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
        while not res or res[0] != 1:
            res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
        if debug:
            print("Moved to home position")
    except NiryoRosWrapperException as e:
        print(f"Error occurred: {e}")
        niryo_robot.clear_collision_detected()
        res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
        if not res or res[0] != 1:
            print("Error moving to home position")

    return res

def push_along_line_from_action(action, debug=False):
    '''
    Pushes the robot along a line, where the action vector defines the push.
    action: [x, y, theta, length] - the parameters learned by the RL agent.
    debug: boolean to print debug statements
    '''

    # Define the real-world limits for each action dimension
    real_x_min, real_x_max = 0.167, 0.432
    real_y_min, real_y_max = -0.132, 0.132
    real_z_min, real_z_max = 0.0, 0.05 

    workspace_length = min(real_x_max - real_x_min, real_y_max - real_y_min)

    real_theta_min, real_theta_max = -180, 180
    real_length_min, real_length_max = 0.1 * workspace_length, 0.5 * workspace_length  # Limit the min and max length proportional to the workspace length
    if debug:
        print(action)

    # Denormalize each action dimension
    x = denormalize_action(action[0], real_x_min, real_x_max) # Range length = max_action_value - min_action_value
    y = denormalize_action(action[1], real_y_min, real_y_max)
    z = denormalize_action(action[2], real_z_min, real_z_max)
    theta = denormalize_action(action[3], real_theta_min, real_theta_max)
    theta_radians = math.radians(theta)
    length = denormalize_action(action[4], real_length_min, real_length_max)
    if debug:
        print(f'Action denormalized: {x,y,z,theta,length}')


    # Go to home position
    res = go_to_home_position()

    # Close the gripper
    res = niryo_robot.grasp_with_tool()
    while not res:
        res = niryo_robot.grasp_with_tool()
    if debug:
        print("Closed the gripper")

    # Move to the starting position
    res = niryo_robot.move_pose(x, y, max(z + 0.07, 0.1), 0.0, 1.57, 0)
    if debug:
        print(f"Moved to the starting position: x={x}, y={y}, z={z}")

    # Calculate the final position based on the length and theta
    final_x, final_y = x + length * math.cos(theta_radians), y + length * math.sin(theta_radians)
    res = niryo_robot.move_pose(final_x, final_y, max(z + 0.07, 0.1), 0.0, 1.57, 0)
    if res[0] != 1:
        if debug:
            print("Error moving to the final position")
        raise NiryoRosWrapperException("Error moving to the final position")
    if debug:
        print(f"Moved to the final position: final_x={final_x}, final_y={final_y}")

    # Return to home position
    res = go_to_home_position()

    return True

def calculate_pixel_clutter_density(rgb_image, depth_image):
    '''
    Calculates the pixel clutter density in the image.
    rgb_image: the RGB image
    depth_image: the depth image
    '''
    # Check if the images are valid
    if rgb_image is None or depth_image is None:
        return None
        
    # Convert the RGB image to grayscale and apply edge detection
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the image to detect objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Initialize the list of objects
    objects = []
    total_area = rgb_image.shape[0] * rgb_image.shape[1]  # Total image area

    for contour in contours:
        # Calculate the bounding box of each object
        x, y, w, h = cv2.boundingRect(contour)
            
        # Remove small objects (noise)
        if w * h < 0.01 * total_area:
            continue

        centroid = (x + w // 2, y + h // 2)

        # Find the corresponding depth of the object by averaging depth pixels within the bounding box
        depth_region = depth_image[y:y+h, x:x+w]
        avg_depth = np.mean(depth_region)

        # Calculate object size (approximated by bounding box area)
        object_size = w * h
            
        # Append object position and depth
        objects.append((centroid, avg_depth, object_size))
        
    # Initialize clutter density map
    clutter_density_map = np.zeros_like(gray, dtype=np.float32)

    # Define window size
    window_size = 5  # Adjust this value as needed

    def calculate_clutter_for_window(x, y):
        clutter_density = 0
        for other_centroid, other_depth, object_size in objects:
            distance = np.linalg.norm(np.array((x, y)) - np.array(other_centroid))
            clutter_density += (1 / (distance + 1e-5)) * object_size
        return clutter_density

    # Calculate clutter density for each window
    for x in range(0, rgb_image.shape[1], window_size):
        for y in range(0, rgb_image.shape[0], window_size):
            clutter_density = min(calculate_clutter_for_window(x, y), 500)  # Clip the clutter density values
            clutter_density_map[y:y+window_size, x:x+window_size] = clutter_density

    # Normalize the clutter density map
    clutter_density_normalized = clutter_density_map / 500
    
    
    total_density = np.sum(clutter_density_normalized)
    # print(" total_density:", total_density)
    if total_density == 0:   ##to tackle if the total_density sum comes 0
        clutter_density_normalized = calculate_pixel_clutter_density(rgb_image, depth_image)

    
    return clutter_density_normalized


class NiryoController:
    def __init__(self, model_path):
        # Parse the arguments
        self.args = parse_args()
        
        logging.info('Connecting to camera...')
        self.cam = RealSenseCamera()
        self.cam.connect()
        self.cam_data = CameraData()
        
    
        # Initilaise the Model for Graspable
        self.grasp_model = Graspable(network_path=self.args.network, force_cpu=self.args.force_cpu)
        self.transformation_matrix = None 
        env = DummyVecEnv([lambda: NiryoRobotEnv()])
        self.model = SAC.load(model_path)
        self.model.set_env(env)
        print("Environment Observation Space:", env.observation_space)
        print("Environment Action Space:", env.action_space)
        
    def get_state(self):
        
        # Ensure both color and depth images are available
        image_bundle = self.cam.get_image_bundle()
        while (image_bundle is None or 'rgb' not in image_bundle or 'aligned_depth' not in image_bundle):  # Wait for valid images
            rospy.loginfo("Waiting for valid images...")
            image_bundle = self.cam.get_image_bundle()
            
        # Get the RGB and depth images
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        depth_unexpanded = image_bundle['unexpanded_depth']
        depth_frame = image_bundle['depth_frame']

        
        while self.transformation_matrix is None:
            # Generate transformation matrix

            self.transformation_matrix = self.grasp_model.aruco_marker_detect(rgb,self.cam.camera_matrix,self.cam.distortion_matrix)
            
            print("Transformation matrix generated.")
            if self.transformation_matrix is not None:
                # Wait for user input to continue
                input("Press Enter to start prediction...")
                break

        # Get the camera data
        x, depth_image, denormalised_depth, rgb_img_ml = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Check if the target object is graspable
        self.graspable = self.grasp_model.run_graspable(x, depth_image, denormalised_depth,self.cam_data.get_rgb(rgb,norm=False))
        
        if(len(self.graspable)!=0):
            res=self.grasp_model.pick(self.graspable,depth_frame,depth_unexpanded,self.transformation_matrix)

        # Get the denormalised color image
        color_image = self.cam_data.get_rgb(rgb, False)

        # Convert to HSV and create mask for blue color
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        # Color range for shades of red
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Create a mask for the red color
        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        mask_image = mask1 | mask2

        # Calculate the number of white pixels in the mask
        white_pixel_count = cv2.countNonZero(mask_image)
   
        # Calculate the centroid of the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image)
        white_pixel_count = stats[:, cv2.CC_STAT_AREA].sum()  # Total white pixel count
        self.centroid = centroids[1] if num_labels > 1 else np.array([-1.0, -1.0], dtype=np.float32)  # Handle invalid case

        # Convert the color image to grayscale and normalise
        gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
        gray_image_normalised = cv2.normalize(gray_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Normalising the grayscaled normalised rgb image
        gray_image_normalised = gray_image_normalised.reshape(gray_image_normalised.shape[0], gray_image_normalised.shape[1], 1)

        # Normalise the depth image
        min_abs, max_abs = 100, 1000
        depth_image = np.clip((denormalised_depth - min_abs) / (max_abs - min_abs), 0, 1)
        depth_image = depth_image.reshape(depth_image.shape[0], depth_image.shape[1], 1)
        print(depth_image.shape)


        # Calculate the pixel clutter density
        self.clutter_map = calculate_pixel_clutter_density(color_image, depth_image)


        # Return the state as a dictionary matching observation space
        state = {
            'gray': gray_image_normalised,
            'depth': depth_image,  # Add channel dimension for depth
            'white_pixel_count': np.array(white_pixel_count, dtype=np.int32),# Send number of white pixels and centroid coordinates
            'centroid':np.array(self.centroid,dtype = np.float32),
            'clutter_density': self.clutter_map
        }

        rospy.loginfo('New STATE registered')

            
        return state
        
        
        
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        print("In control loop")
        while not rospy.is_shutdown():

            
            state = self.get_state()
            state = {key: np.expand_dims(value, axis=0) for key, value in state.items()}
            action, _ = self.model.predict(state)
            print("predicted action is:",action)
            push_along_line_from_action(action[0])
            

            rate.sleep()
 





# Initialize the ROS environment and load SAC model
if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('niryo_rl_node', anonymous=True)

    # Connecting to the ROS Wrapper & calibrating if needed
    niryo_robot = NiryoRosWrapper() # type: ignore
    niryo_robot.calibrate_auto()

    # Update tool
    niryo_robot.update_tool()
    

    model_path = "/home/sanraj/armor_ws/Model/niryo_sac_model_7000_steps.zip"
    robot = NiryoController(model_path)
    print("main")
    robot.run()
    