# armor
ARMOR Robotics


## Niryo Simulation launch command
```bash
roslaunch niryo_robot_bringup desktop_gazebo_simulation.launch
```

### Add the below lines to bashrc for custom objects for niryo simulation
```bash
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:~/your_ros_workspace/src/your_package/models
```
