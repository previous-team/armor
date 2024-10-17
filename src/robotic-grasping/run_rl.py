import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
import cv2
from cv_bridge import CvBridge
import math
from niryo_robot_python_ros_wrapper import *
import random

def push_along_line_from_action(action, z=0.1, debug=False):
    '''
    Pushes the robot along a line, where the action vector defines the push.
    action: [x, y, theta, length] - the parameters learned by the RL agent.
    z: constant z-coordinate for pushing
    debug: boolean to print debug statements
    '''
    x, y, theta, length = action

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


# Define the neural network for the policy (actor) and Q-function (critic)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) #fully connected layer with 256 hidden units
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU() #non-linear activation function
    #Defines how data passes through the network
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) #normalising between -1 and 1 for continuous action space
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        #having two critics helps stabilize training
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4) #learning rate Adam optimizer
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # Soft update for target critics
        self.soft_tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        # Action will have four components: [x, y, theta, length]
        return action.detach().numpy()[0]
    
    def update(self, replay_buffer, batch_size=64, gamma=0.99):
        # Sample a batch of transitions
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        # Update Critic network
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * gamma * target_q

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        critic1_loss = ((q1 - target_q) ** 2).mean() #mean squared error between predicted and target q values
        critic2_loss = ((q2 - target_q) ** 2).mean()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()  #Backpropagation:  gradients of the losses are computed, and the critics are updated
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update Actor network
        actor_loss = -self.critic1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        '''
        actor_loss: The policy gradient loss for the actor is the negative expected Q-value. This encourages the actor to choose actions that maximize the Q-value.
        actor_optimizer updates the actor network's weights.
        '''

        # Soft update for target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data)

# Replay buffer for experience storage
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
    #adds new experience to the buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer) #returns current size of buffer


# Initializing ROS node push along line action
rospy.init_node('niryo_robot_push_along_line')

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
        action_dim = 4  # Example action dimension (joint positions)
        self.sac_agent = SACAgent(state_dim, action_dim)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

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
        # Execute push action using the selected action
        push_along_line_from_action(action, debug=True)
        rospy.sleep(0.1)

        # After pushing, get new state and compute reward
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
    
    '''
    def compute_reward(self, state):
    # Compute the reward based on how close the object is to the target position
    reward = 0
    done = False

    # Example: reward is based on the object’s new position after the push
    object_position = self.get_object_position()  # Assume function to get object’s position
    target_position = self.get_target_position()  # Assume function for target

    distance_to_target = np.linalg.norm(np.array(object_position) - np.array(target_position))
    reward = -distance_to_target  # Negative reward for distance (smaller distance = higher reward)

    # Check if the object is at the target
    if distance_to_target < some_threshold:
        reward += 100  # Bonus reward for reaching the target
        done = True

    return reward, done
    '''

    def target_grasped(self, state):
        # Logic to check if target object is grasped
        return False

    def train(self, num_episodes=100, max_steps_per_episode=1000):
        # Training loop for the SAC agent
        episode_count = 0  # To track episode numbers

        for episode in range(num_episodes):
            state = self.reset()
            episode_reward = 0
            steps_in_episode = 0

            for t in range(max_steps_per_episode):
                # Select push length action from the agent
                push_length = self.sac_agent.select_action(state)

                # Perform the push action
                next_state, reward, done = self.step(push_length)

                # Store transition in replay buffer
                self.replay_buffer.push(state, push_length, reward, next_state, done)

                # Update agent if enough samples are available
                if len(self.replay_buffer) > 64:
                    self.sac_agent.update(self.replay_buffer, batch_size=64)

                state = next_state
                episode_reward += reward
                steps_in_episode += 1

                if done or steps_in_episode >= max_steps_per_episode:
                    break

            # End of episode
            episode_count += 1
            print(f"Episode {episode_count}/{num_episodes}: Total Reward = {episode_reward}, Steps = {steps_in_episode}")

        # Save the model after training
        torch.save(self.sac_agent.actor.state_dict(), "niryo_sac_actor.pth")
        torch.save(self.sac_agent.critic1.state_dict(), "niryo_sac_critic1.pth")
        torch.save(self.sac_agent.critic2.state_dict(), "niryo_sac_critic2.pth")


# Instantiate the environment and train the agent
if __name__ == "__main__":

    # Connecting to the ROS Wrapper & calibrating if needed
    niryo_robot = NiryoRosWrapper()
    niryo_robot.calibrate_auto()

    # Updating tool
    niryo_robot.update_tool()

    env = NiryoRobotRL()
    env.train(num_episodes=100, max_steps_per_episode=1000)

