import rospy
import random
import signal
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

# Initialize the ROS node
rospy.init_node('spawn_and_delete_objects')

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
    {"sdf_file": "src/armor/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned1"},
    {"sdf_file": "src/armor/models/cube_red/model.sdf", "model_name": "cube_red_spawned"},
    {"sdf_file": "src/armor/models/cube_green/model.sdf", "model_name": "cube_green_spawned"},
    {"sdf_file": "src/armor/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned"},
    {"sdf_file": "src/armor/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned"},
    {"sdf_file": "src/armor/models/cube_blue/model.sdf", "model_name": "cube_blue_spawned2"},
    {"sdf_file": "src/armor/models/cube_green/model.sdf", "model_name": "cube_green_spawned2"},
    {"sdf_file": "src/armor/models/cube_green/model.sdf", "model_name": "cube_green_spawned3"},
    {"sdf_file": "src/armor/models/cuboid_blue/model.sdf", "model_name": "cuboid_blue_spawned2"},
    {"sdf_file": "src/armor/models/cuboid_green/model.sdf", "model_name": "cuboid_green_spawned2"}
]

# Function to spawn a model
def spawn_model(sdf_file, model_name, x, y, z):
    with open(sdf_file, "r") as file:
        model_xml = file.read()

    model_pose = Pose()
    model_pose.position.x = x
    model_pose.position.y = y
    model_pose.position.z = z

    resp = spawn_model_prox(model_name, model_xml, "", model_pose, "world")
    print(f"Spawned {model_name}: {resp}")

# Function to delete a model
def delete_model(model_name):
    try:
        resp = delete_model_prox(model_name)
        print(f"Deleted {model_name}: {resp}")
    except rospy.ServiceException as e:
        print(f"Service call failed for {model_name}: {e}")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    rospy.signal_shutdown("User interrupted the program")

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

while not rospy.is_shutdown():
    # Spawn the red cube first
    red_cube = next(model for model in models if model["model_name"] == "cube_red_spawned")
    red_x = random.uniform(0.2, 0.3)  # Randomize x position between 0.2 and 0.3
    red_y = random.uniform(-0.09, 0.09)  # Randomize y position between -0.1 and 0.1
    red_z = random.uniform(0.1, 0.35)  # Randomize z position between 0.1 and 0.3
    spawn_model(red_cube["sdf_file"], red_cube["model_name"], red_x, red_y, red_z)

    # Remove the red cube from the list
    models_without_red = [model for model in models if model["model_name"] != "cube_red_spawned"]

    # Iterate over the models and spawn each one around the red cube
    for model in models_without_red:
        x = random.uniform(red_x - 0.05, red_x + 0.05)  # Randomize x position around the red cube
        y = random.uniform(red_y - 0.05, red_y + 0.05)  # Randomize y position around the red cube
        z = random.uniform(0.1, 0.35)  # Randomize z position between 0.1 and 0.3
        spawn_model(model["sdf_file"], model["model_name"], x, y, z)

    # Wait for 10 seconds
    rospy.sleep(10)

    # Delete all models
    for model in models:
        delete_model(model["model_name"])
        if model["model_name"] != "cube_red_spawned":
            for i in range(1, 4):  # Assuming you have up to 3 instances of each model
                instance_name = f"{model['model_name']}_{i}"
                delete_model(instance_name)