import rospy
import argparse
import numpy as np
import math
import cv2
from niryo_robot_python_ros_wrapper import *
from niryo_robot_utils import NiryoRosWrapperException
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from hardware.cam_gazebo import ROSCameraSubscriber
from utils.data.camera_data_gazebo import CameraData
from run_rl_ml import Graspable
from armor.srv import delete_and_spawn_models

import gym
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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


# Define the custom Niryo environment
class NiryoRobotEnv(gym.Env):
    def __init__(self):
        # Initialize the inherited class
        super(NiryoRobotEnv, self).__init__()

        # Parse the arguments
        self.args = parse_args()

        # Initialize the camera subscriber
        self.cam = ROSCameraSubscriber(
        depth_topic='/camera/depth/image_raw',  
        rgb_topic='/camera/color/image_raw'    
        )

        # Initialize the camera data object
        self.cam_data = CameraData()

        # Define the debug flag. WARNING: Setting to True will print hell lot of debug statements. My system almost ran out of space
        self.debug = False

        # Define variables for the environment
        self.done = False
        self.graspable = None
        self.previous_white_pixel_count = 0
        self.current_white_pixel_count = 0
        self.previous_global_clutter_density = None
        self.curent_global_clutter_density = None
        self.previous_local_clutter_density = None
        self.current_local_clutter_density = None
        self.centroid = np.array([-1.0, -1.0], dtype=np.float32)

        # Episode tracking variables
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_step = 0  # Initialize current step

        # Define the maximum number of steps per episode
        self.max_episode_steps = 40

        # Define radius for local clutter density calculation
        self.local_clutter_radius = 30  # Adjust this value as needed(in pixels)

        # Define image sizes
        img_height, img_width = 224, 224

        # gray image: 1 channel, values range from 0 to 255 which is normalised to 0-1
        gray_low = 0
        gray_high = 1

        # Depth image: 1 channel, values range from 0.1 to x which is normalised to 0-1
        depth_low = 0
        depth_high = 1

        # Mask information: white pixel count (integer), centroid X and Y (floating-point values)
        white_pixel_count_low = 0
        white_pixel_count_high = img_height * img_width  # Maximum possible number of white pixels

        # Mask centroid: X and Y coordinates (floating-point values)
        centroid_low = np.array([0, 0], dtype=np.float32)   # Lower bounds
        centroid_high = np.array([224, 224], dtype=np.float32)

        # Clutter density: 1 channel, values range from 0 to 1
        clutter_density_low = 0
        clutter_density_high = 1

        # Observation space initialization
        self.observation_space = spaces.Dict({
            'gray': spaces.Box(low=gray_low, high=gray_high, shape=(img_height, img_width), dtype=np.uint8), #grayscale image
            'depth': spaces.Box(low=depth_low, high=depth_high, shape=(img_height, img_width), dtype=np.float32),
            'white_pixel_count': spaces.Box(low=white_pixel_count_low, high=white_pixel_count_high, shape=(1,), dtype=np.int32),
            'centroid': spaces.Box(low=centroid_low, high=centroid_high, shape=(2,), dtype=np.float32),  # 2D centroid (X, Y)
            'clutter_density': spaces.Box(low=clutter_density_low, high=clutter_density_high, shape=(img_height, img_width), dtype=np.float32)
        })

        # Initilaise the Model for Graspable
        self.grasp_model = Graspable(network_path=self.args.network, force_cpu=self.args.force_cpu)

        # Define the x, y, and z limits
        xmin_limit = 0.0
        xmax_limit = 1.0
        ymin_limit = 0.0
        ymax_limit = 1.0
        zmin_limit = 0.0
        zmax_limit = 1.0
        thetamin_limit = 0
        thetamax_limit = 1 # scale it from -180 to 180
        lenmin_limit = 0
        lenmax_limit = 1


        # Define the action space: [x, y, z, theta, length]
        # Set specific limits for x, y, z, and uniform limits for theta and length
        low_limits = np.array([xmin_limit, ymin_limit, zmin_limit, thetamin_limit, lenmin_limit])  # Example theta and length range [-1, 1]
        high_limits = np.array([xmax_limit, ymax_limit, zmax_limit, thetamax_limit, lenmax_limit])

        self.action_space = spaces.Box(low=low_limits, high=high_limits,shape = (5,), dtype=np.float32)
        
        self.log_file = open('episode.txt', 'a')

    def reset(self):
        if self.debug:
            print('in reset')
        try:
            # Move robot to home position
            res = go_to_home_position()
            
            state = spaces.Dict()
            # Reset the environment variables
            self.done = False
            self.graspable = None
            self.previous_white_pixel_count = 0
            self.current_white_pixel_count = 0
            self.previous_global_clutter_density = None
            self.current_global_clutter_density = None
            self.previous_local_clutter_density = None
            self.current_local_clutter_density = None
            self.centroid = np.array([-1.0, -1.0], dtype=np.float32)

            # Episode tracking variables
            self.current_episode_reward = 0
            self.current_step = 0  # Initialize current step

            # Reset the environment and return the initial state
            rospy.wait_for_service('/delete_and_spawn_models')
            reset_simulation = rospy.ServiceProxy('/delete_and_spawn_models', delete_and_spawn_models)
            resp = reset_simulation()

            # Check of the target object is graspable
            state = self.get_state()
            
            # If the target object is graspable, reset the simulation
            while self.graspable:
                rospy.loginfo("Resetting simulation as target object is graspable")
                resp = reset_simulation()

                state = self.get_state()

            rospy.sleep(1)  # Wait for the simulation to reset
            
            # Get the initial state
            state = self.get_state()
            
        except Exception as e:
            rospy.logwarn(f"Exception occurred: {e}")
            niryo_robot.clear_collision_detected()
            state = self.get_state()
        return state

    def get_state(self):
        if self.debug:
            print('in state')
    
        # Ensure both color and depth images are available
        image_bundle = self.cam.get_image_bundle()
        while (image_bundle is None or 'rgb' not in image_bundle or 'aligned_depth' not in image_bundle):  # Wait for valid images
            rospy.loginfo("Waiting for valid images...")
            image_bundle = self.cam.get_image_bundle()
        
        # Get the RGB and depth images
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']

        # Get the camera data
        x, depth_image, denormalised_depth, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Check if the target object is graspable
        self.graspable = self.grasp_model.run_graspable(x, depth_image, denormalised_depth, rgb_img )

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
        if self.debug:
            print(f'white pixel count {white_pixel_count}')
        self.previous_white_pixel_count = self.current_white_pixel_count  # Update the previous white pixel count
        self.current_white_pixel_count = white_pixel_count

        # Calculate the centroid of the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image)
        white_pixel_count = stats[:, cv2.CC_STAT_AREA].sum()  # Total white pixel count
        self.centroid = centroids[0] if num_labels > 1 else np.array([-1.0, -1.0], dtype=np.float32)  # Handle invalid case

        # Convert the color image to grayscale and normalise
        gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
        gray_image_normalised = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)  # Normalising the grayscaled normalised rgb image

        # Normalise the depth image
        min_abs, max_abs = 10, 100
        depth_image = np.clip((denormalised_depth - min_abs) / (max_abs - min_abs), 0, 1)
        depth_image = depth_image.squeeze()

        if self.debug:
            print(f'gray{gray_image_normalised.shape}')
            print(f'depth{depth_image.shape}')

        # Calculate the pixel clutter density
        self.clutter_map = calculate_pixel_clutter_density(color_image, depth_image)

        # Calculate the global clutter density
        self.previous_global_clutter_density = self.current_global_clutter_density
        self.current_global_clutter_density = np.sum(self.clutter_map) # changed from mean to sum

        # Calculate the local clutter density
        if self.centroid[0] != -1.0 and self.centroid[1] != -1.0:
            self.previous_local_clutter_density = self.current_local_clutter_density
            self.current_local_clutter_density = np.sum(self.clutter_map[
                min(max(0, int(self.centroid[1] - self.local_clutter_radius)), 224):min(max(0, int(self.centroid[1] + self.local_clutter_radius)), 224), 
                min(max(0, int(self.centroid[0] - self.local_clutter_radius)), 224):min(max(0, int(self.centroid[0] + self.local_clutter_radius)), 224)])

        # Return the state as a dictionary matching observation space
        state = {
            'gray': gray_image_normalised,
            'depth': depth_image,  # Add channel dimension for depth
            'white_pixel_count': np.array(self.current_white_pixel_count, dtype=np.int32),# Send number of white pixels and centroid coordinates
            'centroid':np.array(self.centroid,dtype = np.float32),
            'clutter_density': self.clutter_map
        }

        rospy.loginfo('New STATE registered')

        if self.debug:
            print(f"State: {state}")
        
        return state

    def step(self, action):
        if self.debug:
            print('in step')
        
        # Initialize the state and info
        state = spaces.Dict()
        info = {}

        # Increment the current step and initialize the reward
        self.current_step += 1
        reward = 0

        try:
            # Check if the target object is graspable
            if self.graspable == False:
                # Perform push action using selected action
                proceed = push_along_line_from_action(action)
                rospy.sleep(0.1)

        except Exception as e:
            # Log the exception and clear the collision detected flag
            rospy.logwarn(f"Exception occurred: {e}")
            niryo_robot.clear_collision_detected()

            # Penalize for the collision
            reward -= 2 #changed from 5 to 2
        
        finally:
            # Get the new state
            state = self.get_state()

            # Compute the reward and check if the episode is done
            computed_reward, self.done = self.compute_reward(state)

            # Update the reward
            reward += computed_reward

            # Update current episode reward
            self.current_episode_reward += reward

        
        # Handle episode completion
        if self.done:
            self.episode_count += 1
            # Log episode reward to info for callback
            self.log_file.write(f'{self.episode_count}: Epsiode reward: {self.current_episode_reward} , No of timestep in the episode: {self.current_step} \n')
            self.log_file.flush()  # Flush to ensure the data is written immediately
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_step  # or use the total steps taken in the episode
            }
        print(f'Current episode reward : {self.current_episode_reward}')

        return state, reward, self.done, info


    def compute_reward(self, state):
        if self.debug:
            print('in reward')
        reward = 0.0
        
        if self.debug:
            print("Current_white_pixel_count:", self.current_white_pixel_count)

        # Check if the target object is graspable
        if self.graspable:
            reward += 15.0
            self.done = True
            rospy.loginfo(f"Ending episode as target object is graspable after actions taken by the bot")
        # Reward for varying white pixel count or clutter density if timestep > 1
        if self.current_step > 1:
            # Reward for increasing white pixel count
            if self.current_white_pixel_count > self.previous_white_pixel_count:
                reward += 2.0
            elif self.current_white_pixel_count <= self.previous_white_pixel_count:
                reward += -2.0

            # Reward for varying clutter density
            if self.current_white_pixel_count == 0:
                if self.previous_global_clutter_density and (self.current_global_clutter_density < self.previous_global_clutter_density):
                    reward += 3.0
                elif self.previous_global_clutter_density and (self.current_global_clutter_density >= self.previous_global_clutter_density):
                    reward += -3.0
            else:
                if self.previous_local_clutter_density and (self.current_local_clutter_density < self.previous_local_clutter_density):
                    reward += 3.0
                elif self.previous_local_clutter_density and (self.current_local_clutter_density >= self.previous_local_clutter_density):
                    reward += -3.0
        # Check if the episode has reached the maximum steps
        if self.current_step >= self.max_episode_steps:
            # reward += -5.0
            self.done = True
            rospy.loginfo("Ending episode as maximum steps reached")

        print(f'Reward:  {reward} , Done:  {self.done}')
        return reward, self.done


# Initialize the ROS environment and SAC model
if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('niryo_rl_node', anonymous=True)

    # Connecting to the ROS Wrapper & calibrating if needed
    niryo_robot = NiryoRosWrapper() # type: ignore
    niryo_robot.calibrate_auto()

    # Update tool
    niryo_robot.update_tool()

    # Create an environment instance
    env = DummyVecEnv([lambda: NiryoRobotEnv()])
    # env = Monitor(env, filename=None, allow_early_resets=True)

    logdir = "logs"
    # Set up SAC model with a specified buffer size
    model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=5000, tensorboard_log=logdir)  # Set buffer size here

    # Set up a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./logs/', name_prefix='niryo_sac_model')

    # Set up a TensorBoard callback
    # tensorboard_callback = TensorBoardCallback(log_dir='./logs/tensorboard/')
   
    # Train the model with the callbacks
    total_timesteps = 10000
    #model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, tensorboard_callback])
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC",callback=checkpoint_callback)

    # Save the trained model
    model.save("niryo_sac_model")
    print(f"Training completed for {total_timesteps} time steps")
    