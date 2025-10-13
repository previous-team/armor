# Autonomous robotic manipulation for grasping a target object in cluttered environments

## Introduction

**ARMOR** (Autonomous Robotic Manipulation for Object Retrieval) is an intelligent robotic manipulation framework designed to **grasp a target object in cluttered and occluded environments**.

The system integrates **Deep Reinforcement Learning (DRL)** and **supervised grasp prediction** to enable robots to perform **strategic pushing and grasping** actions that make the target object accessible and graspable, even under heavy clutter.

At its core, ARMOR combines:
- **Soft Actor-Critic (SAC)** reinforcement learning to learn optimal **non-prehensile pushing** strategies in a **continuous action space**, and  
- **GR-ConvNet**, a deep convolutional network that predicts robust **antipodal grasp points** in real time.

A novel **pixel-wise clutter map** is introduced to quantify environmental clutter and guide the robot’s decision-making process. This integration allows the system to **adaptively clear clutter**, **detect grasp opportunities**, and **seamlessly transition** between pushing and grasping behaviors.

Trained in simulation using **ROS Noetic** and **Gazebo 11**, and deployed directly on a **Niryo Ned2 robotic arm** with an **Intel RealSense D415** camera, ARMOR demonstrates robust **sim-to-real transfer** without additional retraining.

> 🧠 TL;DR, ARMOR presents a robust, learning-based approach for autonomous object retrieval in dense, unstructured environments—combining perception, reasoning, and control into one unified system.

## Publication

This work was published in **Autonomous Robots Journal (Springer Nature)**.<br>
DOI: https://doi.org/10.1007/s10514-025-10214-7 <br>
[Click here to read the full paper](https://rdcu.be/eJ6Xi)

## Demo

[![Watch the video](https://img.youtube.com/vi/RbSYFcpxwHw/0.jpg)](https://youtu.be/RbSYFcpxwHw)

## Implementation / Workflow

This section describes the ARMOR implementation and workflow (high-level, implementation-focused) as presented in the paper **“Autonomous Robotic Manipulation for Grasping a Target Object in Cluttered Environments”**.
<details>
<summary>Click to expand</summary>

### Overview
ARMOR combines a supervised grasp predictor (GR-ConvNet) with a continuous-action DRL policy (Soft Actor-Critic, SAC) that learns **non-prehensile pushing** to make a designated target object visible and graspable in cluttered scenes. A novel **pixel-wise clutter map** is introduced and incorporated into the agent state to guide decluttering.

---

### Perception & Preprocessing
- Input: an RGB-D frame (top-down view). Images are cropped/resized to **224 × 224** for all processing.
- Target detection: image → HSV color thresholding to obtain a binary mask for the chosen target color. From the mask:
  - Compute **pixel count** `N` (visibility measure).
  - Compute **centroid** `C` (normalized pixel coordinates). If `N = 0`, centroid set to `(-1, -1)`.
- Depth data is normalized to `[0,1]` using camera min/max thresholds.

---

### Clutter Map (pixel-wise)
- For every pixel, compute a clutter intensity that aggregates contributions from all detected objects:
  - Contribution from object `i` ∝ `size_i / distance_i`.
- Implementation notes:
  - Use edge detection + contouring to estimate object centroids and sizes.
  - Compute clutter score per window (window size = 5) across the depth/image plane.
  - Clip and normalize clutter map to `[0,1]`.
- The clutter map `ρ(x,y)` encodes spatial density and occlusion information and is included in the RL state.

---

### Grasp Prediction & Validation (GR-ConvNet + Graspability Check)
- GR-ConvNet predicts per-pixel grasp quality, angle and width (antipodal grasps).
- For each candidate grasp (center + angle):
  1. Apply color mask check to ensure the grasp corresponds to the selected target.
  2. Deproject grasp center from image → 3D using camera intrinsics:
     - `x_real = (pixel_x - ppx) / fx * depth`
     - `y_real = (pixel_y - ppy) / fy * depth`
     - `z_real = depth`
  3. Construct a grasp rectangle in 3D using gripper length/width and angle; reproject corners to image plane.
  4. Sample depth along the rectangle edges and compare with center depth:
     - If any edge sample depth ≤ center depth → reject (obstruction).
  5. If valid, execute grasp.
- Only when the grasp passes validation is the prehensile action taken.

---

### RL Formulation (SAC continuous policy)
- **MDP**: `(S, A, P, R, γ)`
- **State** `S_t` = `{G, D, N, C, ρ}`:
  - `G` = normalized grayscale image ∈ `[0,1]^{H×W}`  
  - `D` = normalized depth image ∈ `[0,1]^{H×W}`  
  - `N` = target pixel count (visibility scalar)  
  - `C` = centroid `(x_c, y_c)` (normalized; `(-1,-1)` if not visible)  
  - `ρ` = pixel-wise clutter map ∈ `[0,1]^{H×W}`
- **Action** `A_t` (continuous, normalized to `[0,1]` for each component) — 5D vector:
  - `x, y` : normalized start position coordinates (map to workspace)
  - `z` : normalized height (mapped to physical range 0–5 cm)
  - `θ` : normalized angle (mapped to `[-180°, 180°]`)
  - `l` : normalized push length (mapped to `[0.1·d_min, 0.5·d_min]`)
- **Denormalization**: action_denorm = action_min + (action_max − action_min) * action_norm
  - Final push end position computed via:
    - `x_final = x_start + length * cos(θ)`
    - `y_final = y_start + length * sin(θ)`

---

### Reward Design
Combined, dense reward components encourage visibility, clutter reduction, and safe behavior:
- **Graspability reward**: positive reward when the target becomes graspable.
- **Visibility reward**: positive if `N_t` increases vs `N_{t-1}`; negative if it decreases.
- **Clutter reward**:
  - When target invisible: reward based on decrease in **global** mean clutter.
  - When target visible: reward based on decrease in **local** mean clutter around centroid `C`.
- **Collision penalty**: negative reward for collisions or unsafe pushes.
- Rewards are carefully balanced to encourage decluttering that leads to successful grasping.

---

### Training & Evaluation
- Training uses **Soft Actor-Critic (SAC)** with replay buffer and entropy regularization.
- Models trained for **10,000 timesteps** in randomized scenes (varying object counts/shapes/positions).
- Variants trained:
  - With vs without pixel-wise clutter map in state.
  - Different discount factors: `γ = 0.99` and `γ = 0.90`.
- Key logged metrics:
  - Mean episode length (steps to complete)
  - Mean episode reward
  - Actor / Critic loss curves
- Evaluation metrics:
  - **Completion Rate (C%)** — fraction of runs where the target is grasped.
  - **Motion Number (MN)** — average number of pushes needed to make object graspable.
  - Grasp success (GS%) reported via GR-ConvNet accuracy and grasp validation.

---

### Simulation & Real-World Transfer
- **Simulation**: Gazebo + randomized spawn service to generate diverse cluttered scenes (10–50 objects). Top-down Intel RealSense camera simulated.
- **Hardware**:
  - Niryo NED2 6-DOF manipulator
  - Intel RealSense D415 RGB-D camera (overhead)
  - ArUco marker for camera ↔ robot frame transform calibration
- The policy outputs normalized actions; a workspace-based denormalization allows direct transfer from sim to real **without additional retraining**.

---

### Experimental Observations (summary)
- Including the **pixel-wise clutter map** improves convergence speed, reduces mean episode length, and increases average reward.
- A discount factor of **γ = 0.90** produced faster initial convergence (shorter episodes), while **γ = 0.99** tended to converge more slowly but could stabilize to different long-term behaviors.
- Continuous action SAC provides finer pushing control compared to fixed discrete push orientations/lengths, at the cost of sometimes higher motion counts (i.e., more pushes to reach graspable state) but higher grasp success and completion rates overall.

---

### Practical Notes & Constraints
- Grasp validation uses local depth comparisons to avoid collisions — the final decision to grasp is conservative to ensure reliable physical execution.
- Action ranges are bounded to the camera-visible workspace to prevent pushes that move objects outside the observable region.
- The `pixel count` (N) is used instead of full segmentation masks in the RL state to improve generalization across object shapes and colors.

---

This implementation workflow captures the perception-to-action pipeline, the RL formulation, the clutter-aware state design, and sim-to-real considerations as described in the paper.

</details>

## Running the Project

### Requirements
- **Ubuntu 20.04** (or compatible version)
- **ROS Noetic** (or compatible version)
- **Gazebo** (for simulation)
- **Python 3** (for Q-learning implementation)

To run the ARMOR project, follow these steps:
1. Clone the repository in the `src` directory of your catkin workspace using the following command
```bash
git clone https://github.com/Space-Gamer/armor-robotics.git
```
2. Navigate to the root of your catkin workspace and build the project using the following commands
```bash
catkin_make
source devel/setup.bash
```
3. Install the required dependencies (virtual environment recommended) using the following command
```bash
pip install -r requirements.txt
```
4. Launch the Gazebo simulation environment with the following command
```bash
roslaunch armor simple_box.launch
```
5. In a new terminal, run the main ARMOR Training script using the following command
```bash
python3 src/robotic-grasping/run_rl_sac_with_clutter.py
```
6. You can evaluate the trained model using the following command
```bash
python3 src/robotic-grasping/evaluate_rl_sac_with_clutter.py <path_to_trained_model>
```

__Note__: Add the below lines to bashrc for custom objects for niryo simulation
```bash
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:~/<your workspace>/src/armor/models
```