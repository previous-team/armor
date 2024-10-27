import rospy
import argparse
from std_msgs.msg import Bool
import numpy as np
import torch
import math
import cv2
from niryo_robot_python_ros_wrapper import *
import random
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from hardware.cam_gazebo import ROSCameraSubscriber
from utils.data.camera_data_gazebo import CameraData
from run_rl_ml import Graspable
from armor.srv import delete_and_spawn_models

import gym
from gym import spaces
from std_srvs.srv import Empty
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='src/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_17_iou_0.96',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


def denormalize_action(action, action_min, action_max):
    return action_min + (action_max - action_min) * action

def push_along_line_from_action(action, z=0.1, debug=False):
    '''
    Pushes the robot along a line, where the action vector defines the push.
    action: [x, y, theta, length] - the parameters learned by the RL agent.
    z: constant z-coordinate for pushing
    debug: boolean to print debug statements
    '''

    # real_x_min, real_x_max = 0.173, 0.427
    # real_y_min, real_y_max = -0.124, 0.124  
    # real_z_min, real_z_max = 0.0, 0.2  

    real_x_min, real_x_max = 0.167, 0.432
    real_y_min, real_y_max = -0.132, 0.132  
    real_z_min, real_z_max = 0.0, 0.1 
    real_theta_min, real_theta_max = -180,180
    real_length_min, real_length_max = 0.0,(0.158*0.6) # *0.6 of workspace  #antipodal paper reference 

    # Normalized action values
    # Clipping the normalized action values to ensure they are within range
    norm_x, norm_y, norm_z, norm_theta, norm_length = np.clip(action, [0, -1, 0, -1, 0], [1, 1, 1, 1, 1])
    print(f'Action normalized: {norm_x,norm_y,norm_z,norm_theta,norm_length}')

    # Denormalize each action dimension
    x = denormalize_action(norm_x, real_x_min, real_x_max)
    y = denormalize_action(norm_y, real_y_min, real_y_max)
    z = denormalize_action(norm_z, real_z_min, real_z_max)
    theta = denormalize_action(norm_theta, real_theta_min, real_theta_max)
    theta = np.clip(theta,-180,180)
    theta_radians = math.radians(theta)
    length = denormalize_action(norm_length, real_length_min, real_length_max)
    print(f'Action denormalized: {x,y,z,theta,length}')
    # Go to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
    if debug and res and res[0] == 1:
        print("Moved to home position")

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
    if debug:
        print(f"Moved to the final position: final_x={final_x}, final_y={final_y}")

    # Return to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
    if debug:
        print("Returned to home position")

    return True

def check_state_values(state):
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            # If the array is non-empty and has truthy values
            if not np.any(value):
                rospy.loginfo(f"Array in key '{key}' is falsy or empty.")
                return False
        else:
            # If value is not an array, check it directly
            if not value:
                rospy.loginfo(f"Value for key '{key}' is falsy.")
                return False
    return True


# Define the custom Niryo environment
class NiryoRobotEnv(gym.Env):
    def __init__(self):
        super(NiryoRobotEnv, self).__init__()
        self.args = parse_args()
        self.cam = ROSCameraSubscriber(
        depth_topic='/camera/depth/image_raw',  
        rgb_topic='/camera/color/image_raw'    
        )
        self.cam_data = CameraData(include_depth=self.args.use_depth, include_rgb=self.args.use_rgb)
        # Define image sizes
        img_height, img_width = 224, 224 

        # gray image: 1 channel, values range from 0 to 255
        gray_low = 0
        gray_high = 1

        # Depth image: 1 channel, values range from 0 to 0.5
        depth_low = 0
        depth_high = 1 # TODO check the normalised depth values

        # Mask information: white pixel count (integer), centroid X and Y (floating-point values)
        white_pixel_count_low = 0
        white_pixel_count_high = img_height * img_width  # Maximum possible number of white pixels

        centroid_low = np.array([0, 0], dtype=np.float32)   # Lower bounds
        centroid_high = np.array([224, 224], dtype=np.float32)

        # Observation space initialization
        self.observation_space = spaces.Dict({
            'gray': spaces.Box(low=gray_low, high=gray_high, shape=(img_height, img_width), dtype=np.uint8), #grayscale image
            'depth': spaces.Box(low=depth_low, high=depth_high, shape=(img_height, img_width,1), dtype=np.float32),
            'white_pixel_count': spaces.Box(low=white_pixel_count_low, high=white_pixel_count_high, shape=(1,), dtype=np.int32),
            'centroid': spaces.Box(low=centroid_low, high=centroid_high, shape=(2,), dtype=np.float32)  # 2D centroid (X, Y)
        })

        # Define the x, y, and z limits
        xmin_limit = 0.0
        xmax_limit = 1.0
        ymin_limit = -1.0
        ymax_limit = 1.0
        zmin_limit = 0.0
        zmax_limit = 1.0
        thetamin_limit = -1
        thetamax_limit = 1 # scale it from 0-360
        lenmin_limit = 0
        lenmax_limit = 1


        # Define the action space: [x, y, z, theta, length]
        # Set specific limits for x, y, z, and uniform limits for theta and length
        low_limits = np.array([xmin_limit, ymin_limit, zmin_limit, thetamin_limit, lenmin_limit])  # Example theta and length range [-1, 1]
        high_limits = np.array([xmax_limit, ymax_limit, zmax_limit, thetamax_limit, lenmax_limit])

        self.action_space = spaces.Box(low=low_limits, high=high_limits,shape = (5,), dtype=np.float32)

    def reset(self):
        print('in reset')
        try:
            state = spaces.Dict()
            self.previous_white_pixel_count = 0
            # Episode tracking variables
            self.current_episode_reward = 0
            self.episode_count = 0
            self.current_step = 0  # Initialize current step

            
            self.done = False
            self.graspable = None
            # self.centroid = np.array([-1.0, -1.0], dtype=np.float32)


            self.current_episode_reward = 0
            self.current_step = 0 

                #parameters to tak careof collision conditions
            self.collision_count = 0  
            self.max_collisions = 5

            self.current_pixel_count=0

            # Reset the environment and return the initial state
            rospy.wait_for_service('/delete_and_spawn_models')
            reset_simulation = rospy.ServiceProxy('/delete_and_spawn_models', delete_and_spawn_models)
            resp = reset_simulation()

            rospy.sleep(0.3)

            # Return to home position
            res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
            
            state = self.get_state()
            while state is None:
                state = self.get_state()
                if state is None:
                    rospy.loginfo("Waiting for valid state...")
            
           
        except:
            niryo_robot.clear_collision_detected()
            state = self.get_state()
            while state is None:
                state = self.get_state()
                if state is None:
                    rospy.loginfo("Waiting for valid state...")
        return state

    def get_state(self):
        print('in state')
        # Ensure both color and depth images are available
        image_bundle = self.cam.get_image_bundle()
        if image_bundle is None or 'rgb' not in image_bundle or 'aligned_depth' not in image_bundle:
            rospy.loginfo("Waiting for valid images...")
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x,depth_image, denormalised_depth, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
        obj = Graspable(network_path=self.args.network,force_cpu=False)
        self.graspable = obj.run_graspable(x,depth_image, denormalised_depth, rgb_img )
        color_image = self.cam_data.get_rgb(rgb,False) # denormalised rgb image for masking of target object



        # # Convert to HSV and create mask for blue color
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        # lower_blue = np.array([100, 150, 50])
        # upper_blue = np.array([140, 255, 255])
        # mask_image = cv2.inRange(hsv_image, lower_blue, upper_blue)
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        rospy.loginfo(f'New STATE registered')
        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        mask_image = mask1 | mask2

        # Calculate the number of white pixels in the mask
        white_pixel_count = cv2.countNonZero(mask_image)
        print(f'white pixel count {white_pixel_count}')
        self.current_pixel_count=white_pixel_count

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image)  #another faster approach for centroid calculation in a binary image
        white_pixel_count = stats[:, cv2.CC_STAT_AREA].sum()  # Total white pixel count
        # print(f'num_labels{num_labels}')
        self.centroid = centroids[0] if num_labels > 1 else np.array([-1.0, -1.0], dtype=np.float32)  # Handle invalid case

        # # Calculate the centroid of the masked area (if there are white pixels)
        # M = cv2.moments(mask_image)
        # if M["m00"] > 0:  # Ensure there are white pixels
        #     cX = int(M["m10"] / M["m00"])  # X coordinate of the centroid
        #     cY = int(M["m01"] / M["m00"])  # Y coordinate of the centroid
        # else:
        #     cX, cY = -1, -1  # If no white pixels, set centroid to an invalid value
        # print(f'Centroid{self.centroid}')

        print(f'color image shape{color_image.shape}')

        gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
        gray_image_normalised = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX) #normalising the grayscaled normalised rgb image
            
        # gray_image_normalised = np.expand_dims(gray_image_normalised, axis=-1)
        # gray_image_normalised = gray_image_normalised.reshape(224,224,1)
        
        # Add an extra dimension to the depth image to match the expected (224, 224, 1) shape

        depth_image = depth_image.transpose((1,2,0))
        print(f'gray{gray_image_normalised.shape}')
        print(f'depth{depth_image.shape}')
        # Return the state as a dictionary matching observation space
        state = {
            'gray': gray_image_normalised,
            'depth': depth_image,  # Add channel dimension for depth
            'white_pixel_count': np.array(self.current_pixel_count, dtype=np.int32),# Send number of white pixels and centroid coordinates
            'centroid':np.array(self.centroid,dtype = np.float32)
        }
        print(f"State: {state}")
        
        return state

    def step(self, action):
        state = spaces.Dict()
        info = {}
        # state = self.get_state()
        # while state is None:
        #     state = self.get_state()
        #     if state is None:
        #         rospy.loginfo("Waiting for valid state...")
        print('in step')
        try:
            # Increment the current step
            self.current_step += 1
            reward = 0 
            # After pushing, get new state and compute reward
            
            if self.graspable == False:
                # Perform push action using selected action
                proceed = push_along_line_from_action(action) #put it after reward is computed TODO
                rospy.sleep(0.1)
            state = self.get_state()
            while state is None:
                state = self.get_state()
                if state is None:
                    rospy.loginfo("Waiting for valid state...")

            reward,self.done = self.compute_reward(state)
            # Update current episode reward
            self.current_episode_reward += reward
        except Exception as e:      # TODO NEGATIVE REWARD FOR COLLISION
            rospy.logwarn(f"Exception occurred: {e}")
            niryo_robot.clear_collision_detected()

            state = self.get_state()
            while state is None:
                state = self.get_state()
                if state is None:
                    rospy.loginfo("Waiting for valid state...")

            # Penalize for the collision
            self.current_episode_reward -= 20
            self.collision_count += 1

            # Check if the number of collisions exceeds the maximum allowed
            if self.collision_count >= self.max_collisions:
                self.done = True
                rospy.logwarn(f"Maximum collision limit reached ({self.max_collisions}). Ending episode.")
            else:
                self.done = False  # continue even after collision

            print(f"Collision detected. Total collisions: {self.collision_count}")

        
        # Handle episode completion
        if self.done:
            self.episode_count += 1
            # Log episode reward to info for callback
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_step  # or use the total steps taken in the episode
            }
        print(f'Current episode reward : {self.current_episode_reward}')

        return state, self.current_episode_reward,self.done, info


    def compute_reward(self, state):
        print('in reward')
        reward = 0.0
        
        # Retrieve the current white pixel count from the state
        current_white_pixel_count = self.current_pixel_count
        # print("current:",current_white_pixel_count)
        
        # Check if the target is grasped
        if self.graspable:
            reward += 10.0
            self.done = True
            rospy.loginfo(f"Ending episode as target object is graspable")
        else:
            reward += -1.0
            
    
        # Reward for increasing white pixel count
        if current_white_pixel_count != self.previous_white_pixel_count:
            reward += (current_white_pixel_count - self.previous_white_pixel_count)/100.0  # You can adjust the reward value as needed # TODO CHANGE REWARD
        else:
            reward += -1.0 #just for avoiding taking action that dont really help increase graspability of target object

        # Update the previous white pixel count for the next call
        self.previous_white_pixel_count = current_white_pixel_count
        print(f'Reward:  {reward} , Done:  {self.done}')
        return reward, self.done

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super(TensorBoardCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        # Log episode rewards and lengths
        if 'episode' in self.locals:
            episode = self.locals['episode']
            self.writer.add_scalar("episode/reward", episode['r'], self.num_timesteps)
            self.writer.add_scalar("episode/length", episode['l'], self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
    

# Initialize the ROS environment and SAC model
if __name__ == "__main__":
   # Connecting to the ROS Wrapper & calibrating if needed
    niryo_robot = NiryoRosWrapper()
    # niryo_robot.calibrate_auto()

    # Update tool
    niryo_robot.update_tool()

    

    # Initialize ROS node
    rospy.init_node('niryo_rl_node', anonymous=True)

    # Return to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)

    # Create an environment instance
    env = DummyVecEnv([lambda: NiryoRobotEnv()])

    # Set up SAC model with a specified buffer size
    model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=5000)  # Set buffer size here

    # Set up a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                            name_prefix='niryo_sac_model')

    # Set up a TensorBoard callback
    tensorboard_callback = TensorBoardCallback(log_dir='./logs/tensorboard/')
   
    # Train the model with the callbacks
    model.learn(total_timesteps=100000, callback=[checkpoint_callback, tensorboard_callback])

    # Save the trained model
    model.save("niryo_sac_model")
    print("Congratulations!!!!! Training done, live long and prosper ;)")
    # done = True
    # # After training, you can evaluate or use the trained model:
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()