#!/usr/bin/env python3

import rospy
import rospkg
import random
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from armor.srv import delete_and_spawn_models, delete_and_spawn_modelsResponse
import tf.transformations as tf_trans

# Initialize the ROS node
rospy.init_node('delete_and_spawn_objects')

# Get the path to the package
rospack = rospkg.RosPack()
package_path = rospack.get_path('armor')

# Wait for the spawn and delete services to be available
rospy.wait_for_service('/gazebo/spawn_sdf_model')
rospy.wait_for_service('/gazebo/delete_model')

try:
    spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
except rospy.ServiceException:
    print("Service call failed")

# List of SDF files and corresponding model names
models = [
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned1"},
    {"sdf_file": f"{package_path}/models/cube_red/model.sdf", "model_name": "cube_red_spawned"},
    {"sdf_file": f"{package_path}/models/cube_green/model.sdf", "model_name": "cube_green_spawned"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned2"},
    {"sdf_file": f"{package_path}/models/cube_green/model.sdf", "model_name": "cube_green_spawned2"},
    {"sdf_file": f"{package_path}/models/cube_green/model.sdf", "model_name": "cube_green_spawned3"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned2"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned2"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned3"},
    {"sdf_file": f"{package_path}/models/cube_green/model.sdf", "model_name": "cube_green_spawned4"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned3"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned3"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned4"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned4"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned4"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned5"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned6"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned5"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned5"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned5"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned6"},
    {"sdf_file": f"{package_path}/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned7"},
    {"sdf_file": f"{package_path}/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned6"},
    {"sdf_file": f"{package_path}/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned6"}
]

# Function to spawn a model
def spawn_model(sdf_file, model_name, x, y, z, roll, pitch, yaw):
    with open(sdf_file, "r") as file:
        model_xml = file.read()

    model_pose = Pose()
    model_pose.position.x = x
    model_pose.position.y = y
    model_pose.position.z = z

    # Convert roll, pitch, yaw to quaternion
    quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
    model_pose.orientation.x = quaternion[0]
    model_pose.orientation.y = quaternion[1]
    model_pose.orientation.z = quaternion[2]
    model_pose.orientation.w = quaternion[3]

    resp = spawn_model_prox(model_name, model_xml, "", model_pose, "world")
    print(f"Spawned {model_name}: {resp}")

# Function to delete a model
def delete_model(model_name):
    try:
        resp = delete_model_prox(model_name)
        print(f"Deleted {model_name}: {resp}")
    except rospy.ServiceException as e:
        print(f"Service call failed for {model_name}: {e}")

# Service callback function
def handle_delete_and_spawn(req):
    try:
        while True:
        # Delete all models
            for model in models:
                delete_model(model["model_name"])

            # Always include the red cube
            red_cube = [model for model in models if model["model_name"] == "cube_red_spawned"][0]
            models_to_spawn = [red_cube]

            # Randomize the number of additional models to spawn (between 10 and the max number of models - 1)
            # num_additional_models_to_spawn = random.randint(10, len(models) - 1)
            num_additional_models_to_spawn = 25
            
            # Randomly select additional models to spawn
            additional_models_to_spawn = random.sample([model for model in models if model["model_name"] != "cube_red_spawned"], num_additional_models_to_spawn)
            
            # Combine the red cube with the additional models
            models_to_spawn.extend(additional_models_to_spawn)

            # Iterate over the models to spawn and spawn each one at a random position and orientation
            for model in models_to_spawn:
                # x = random.uniform(0.17, 0.42)  # Randomize x position between 0.17 and 0.42
                # y = random.uniform(-0.12, 0.12)  # Randomize y position between -0.12 and 0.12
                # z = random.uniform(0.2, 0.4)  # Randomize z position between 0.2 and 0.4
                x = 0.29 # Randomize x position between 0.17 and 0.42
                y = 0.0  # Randomize y position between -0.12 and 0.12
                z = 0.1  # Randomize z position between 0.2 and 0.4
                roll = 0  # Roll is 0
                pitch = 0  # Pitch is 0
                # yaw = random.uniform(0, 2 * 3.14159)  # Randomize yaw between 0 and 2*pi
                yaw = 4  # Randomize yaw between 0 and 2*pi
                spawn_model(model["sdf_file"], model["model_name"], x, y, z, roll, pitch, yaw)

            rospy.sleep(1) # Wait for the models to spawn
            # Check if the red cube is within the workspace boundaries
            red_cube_pose = rospy.wait_for_message(f'/gazebo/model_states', ModelStates)
            red_cube_index = red_cube_pose.name.index("cube_red_spawned")
            red_cube_position = red_cube_pose.pose[red_cube_index].position

            if 0.17 <= red_cube_position.x <= 0.42 and -0.12 <= red_cube_position.y <= 0.12:
                break  # Red cube is within the workspace, exit the loop


        return delete_and_spawn_modelsResponse(status="Success")
    except Exception as e:
        return delete_and_spawn_modelsResponse(status=f"Failed: {e}")

# Create the service
service = rospy.Service('delete_and_spawn_models', delete_and_spawn_models, handle_delete_and_spawn)

# Keep the node running
rospy.spin()