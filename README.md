# armor
ARMOR Robotics


## Niryo Simulation launch command
```bash
roslaunch niryo_robot_bringup desktop_gazebo_simulation.launch
```

### Add the below lines to bashrc for custom objects for niryo simulation
```bash
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:~/<your workspace>/src/armor/models
```
### Give Bash command like this to select world
```bash
roslaunch armor simple_box.launch world_name:=stack
```
