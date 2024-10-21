import rospy
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

import gym
from gym import spaces
from std_srvs.srv import Empty
from torch.utils.tensorboard import SummaryWriter


def push_along_line_from_action(action, z=0.1, debug=False):
    '''
    Pushes the robot along a line, where the action vector defines the push.
    action: [x, y, theta, length] - the parameters learned by the RL agent.
    z: constant z-coordinate for pushing
    debug: boolean to print debug statements
    '''
    x, y, z, theta, length = action
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
    final_x, final_y = x + length * math.cos(theta), y + length * math.sin(theta)
    res = niryo_robot.move_pose(final_x, final_y, max(z + 0.07, 0.1), 0.0, 1.57, 0)
    if debug:
        print(f"Moved to the final position: final_x={final_x}, final_y={final_y}")

    # Return to home position
    res = niryo_robot.move_joints(0, 0.5, -1.25, 0, 0, 0)
    if debug:
        print("Returned to home position")


# Define the custom Niryo environment
class NiryoRobotEnv(gym.Env):
    def __init__(self):
        super(NiryoRobotEnv, self).__init__()
        
        # Define observation and action spaces
        self.color_image_topic = '/camera/color/image_raw'
        self.depth_image_topic = '/camera/depth/image_raw'
        
        # CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera image topics
        rospy.Subscriber(self.color_image_topic, Image, self.color_image_callback)
        rospy.Subscriber(self.depth_image_topic, Image, self.depth_image_callback)

        # Initialize color and depth image variables
        self.color_image = None
        self.depth_image = None

        # Define image sizes
        img_height, img_width = 224, 224

        # RGB image: 3 channels, values range from 0 to 255
        rgb_low = 0
        rgb_high = 255

        # Depth image: 1 channel, values range from 0 to 0.5
        depth_low = 0.0
        depth_high = 0.1  # in meters

        # Mask information: white pixel count (integer), centroid X and Y (floating-point values)
        white_pixel_count_low = 0
        white_pixel_count_high = img_height * img_width  # Maximum possible number of white pixels

        centroid_low = 0  # Centroid coordinates range from 0 to image width/height
        centroid_high = img_height  # Centroid is constrained by image dimensions

        # Observation space initialization
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=rgb_low, high=rgb_high, shape=(img_height, img_width, 3), dtype=np.uint8),
            'depth': spaces.Box(low=depth_low, high=depth_high, shape=(img_height, img_width, 1), dtype=np.float32),
            'white_pixel_count': spaces.Box(low=white_pixel_count_low, high=white_pixel_count_high, shape=(), dtype=np.int32),
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

        self.action_space = spaces.Box(low=low_limits, high=high_limits, dtype=np.float32)

        self.previous_white_pixel_count = 0
        # Episode tracking variables
        self.current_episode_reward = 0
        self.episode_count = 0
        self.current_step = 0  # Initialize current step

        self.done = False
            

    def color_image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def reset(self):
        self.current_episode_reward = 0
        self.current_step = 0 
        # Reset the environment and return the initial state
        rospy.wait_for_service('/gazebo/reset_simulation')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_simulation()
        # TODO arjun's code  #################################################################################################################################
        state = self.get_state()
        return state

    def get_state(self):
        # Ensure both color and depth images are available
        while self.color_image is None or self.depth_image is None:
            rospy.sleep(0.1)

        # Convert to HSV and create mask for blue color
        hsv_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Resize all images to 224x224
        color_resized = cv2.resize(self.color_image, (224, 224))
        depth_resized = cv2.resize(self.depth_image, (224, 224))
        mask_image = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Calculate the number of white pixels in the mask
        white_pixel_count = cv2.countNonZero(mask_image)
        
        # Calculate the centroid of the masked area (if there are white pixels)
        M = cv2.moments(mask_image)
        if M["m00"] > 0:  # Ensure there are white pixels
            cX = int(M["m10"] / M["m00"])  # X coordinate of the centroid
            cY = int(M["m01"] / M["m00"])  # Y coordinate of the centroid
        else:
            cX, cY = -1, -1  # If no white pixels, set centroid to an invalid value
        
        # Return the state as a dictionary matching observation space
        state = {
            'rgb': color_resized,
            'depth': np.expand_dims(depth_resized, axis=-1),  # Add channel dimension for depth
            'white_pixel_count': white_pixel_count, # Send number of white pixels and centroid coordinates
            'centroid':[cX,cY]       
        }
        
        return state


    def step(self, action):

        # Increment the current step
        self.current_step += 1

        # Perform push action using selected action
        push_along_line_from_action(action)
        rospy.sleep(0.1)

        # After pushing, get new state and compute reward
        state = self.get_state()
        reward,self.done = self.compute_reward(state)
        # Update current episode reward
        self.current_episode_reward += reward

        # Handle episode completion
        if self.done:
            self.episode_count += 1
            # Log episode reward to info for callback
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_step  # or use the total steps taken in the episode
            }

        return state, reward,self.done, {}

    def compute_reward(self, state):
        # Custom reward function based on the robot's task
        reward = 0
        
        # Retrieve the current white pixel count from the state
        current_white_pixel_count = state['white_pixel_count']
        
        # Check if the target is grasped
        if self.target_grasped(state):
            reward = 1
            self.done = True

        # Reward for increasing white pixel count
        if current_white_pixel_count > self.previous_white_pixel_count:
            reward = (current_white_pixel_count - self.previous_white_pixel_count)/1000  # You can adjust the reward value as needed

        # Update the previous white pixel count for the next call
        self.previous_white_pixel_count = current_white_pixel_count

        return reward, self.done

    def target_grasped(self, state):
        # Logic to check if the target object is grasped########################################################################################
        return False

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

    # Create an environment instance
    env = DummyVecEnv([lambda: NiryoRobotEnv()])

    # Set up SAC model with a specified buffer size
    model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=10000)  # Set buffer size here

    # Set up a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                            name_prefix='niryo_sac_model')

    # Set up a TensorBoard callback
    tensorboard_callback = TensorBoardCallback(log_dir='./logs/tensorboard/')

    # Train the model with the callbacks
    model.learn(total_timesteps=10000, callback=[checkpoint_callback, tensorboard_callback])

    # Save the trained model
    model.save("niryo_sac_model")
    done = True
    # After training, you can evaluate or use the trained model:
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()