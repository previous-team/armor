import rospy
import numpy as np
import torch
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
import cv2
from cv_bridge import CvBridge
import my_sac_algo as a

class NiryoRobotRL:
    def __init__(self):
        rospy.init_node('niryo_rl_node', anonymous=True)

        # Image topics (replace with actual topics for your camera)
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

        # Initialize SAC agent (custom SAC implementation)
        state_dim = 224 * 224 * 2  # Example state dimension, flattened color + depth images
        action_dim = 6  # Example action dimension (joint positions)
        self.sac_agent = a.SACAgent(state_dim, action_dim)

        # Replay buffer
        self.replay_buffer = a.ReplayBuffer()

    def color_image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def reset(self):
        # Reset simulation
        rospy.wait_for_service('/gazebo/reset_simulation')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_simulation()

        # Return initial state
        state = self.get_state()
        return state

    def get_state(self):
        # Ensure both color and depth images are available
        while self.color_image is None or self.depth_image is None:
            rospy.sleep(0.1)

        # Resize images for consistency
        color_resized = cv2.resize(self.color_image, (224, 224))
        depth_resized = cv2.resize(self.depth_image, (224, 224))

        # Flatten and concatenate color and depth images
        state = np.concatenate((color_resized.flatten(), depth_resized.flatten()))
        return state

    def step(self, action):
        # Publish the action (joint angles)
        # TODO actions still need to be defined
        joint_msg = JointState()
        joint_msg.position = action
        self.joint_pub.publish(joint_msg)

        rospy.sleep(0.1)

        # Get new state and compute reward
        state = self.get_state()
        reward, done = self.compute_reward(state)

        return state, reward, done

    def compute_reward(self, state):
        # Custom reward function based on the robot's task
        reward = 0
        done = False
        # Example: reward based on whether the target is grasped
        #  TODO REWARDS FOR DIFFERENT ACTIONS
        if self.target_grasped(state):
            reward = 100
            done = True
        return reward, done

    def target_grasped(self, state):
        # Logic to check if target object is grasped
        return False

    def train(self, total_timesteps=100000):
        # Training loop for the SAC agent
        state = self.reset()
        for t in range(total_timesteps):
            # Select action from the agent
            action = self.sac_agent.select_action(state)
            
            # Perform action in environment
            next_state, reward, done = self.step(action)

            # Store transition in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            if len(self.replay_buffer) > 64:
                self.sac_agent.update(self.replay_buffer, batch_size=64)

            state = next_state

            if done:
                state = self.reset()

        # Save the model after training
        torch.save(self.sac_agent.actor.state_dict(), "niryo_sac_actor.pth")
        torch.save(self.sac_agent.critic1.state_dict(), "niryo_sac_critic1.pth")
        torch.save(self.sac_agent.critic2.state_dict(), "niryo_sac_critic2.pth")


# Instantiate the environment and train the agent
if __name__ == "__main__":
    env = NiryoRobotRL()
    env.train(total_timesteps=100000)
