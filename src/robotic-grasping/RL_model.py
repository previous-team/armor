import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from stable_baselines3 import SAC

class NiryoRobotRL:
    def __init__(self):
        rospy.init_node('niryo_rl_node', anonymous=True)

        # Publisher for sending joint commands
        self.joint_pub = rospy.Publisher('/robot_joint_position_controller/command', JointState, queue_size=10)
        rospy.wait_for_message('/joint_states', JointState)

    def reset(self):
        # Reset simulation
        rospy.wait_for_service('/gazebo/reset_simulation')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_simulation()

        # Return initial state
        state = self.get_state()
        return state

    def get_state(self):
        # Get current state (e.g., joint angles, object positions, vision data)
        joint_state = rospy.wait_for_message('/joint_states', JointState)
        state = np.array(joint_state.position) 
        return state

    def step(self, action):
        # Publish the action (joint angles)
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
        if self.target_grasped(state):
            reward = 100
            done = True
        return reward, done

    def target_grasped(self, state):
        # Logic to check if target object is grasped
        return False

# Instantiate the environment
env = NiryoRobotRL()

# Use Stable Baselines 3 with the custom environment
model = SAC('MlpPolicy', env, verbose=1)

# Train the model for a specified number of timesteps
model.learn(total_timesteps=100000)

# Save the model
model.save("niryo_sac_model")
