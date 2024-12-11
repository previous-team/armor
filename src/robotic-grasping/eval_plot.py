#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

# Initialize variables
episode_numbers = []
percent_negative_rewards = []

# Read the file
with open("model_eval.txt", "r") as file:
    lines = file.readlines()

# Parse the file
current_episode = None
negative_rewards = 0
total_actions = 0

for line in lines:
    # Match End of Episode line to get episode number and total actions
    match_episode = re.search(r"End of Episode: (\d+).*No of actions taken: (\d+)", line)
    if match_episode:
        if current_episode is not None:
            # Calculate percentage for the previous episode
            percent_negative_rewards.append((negative_rewards / total_actions) * 100)
        
        # Update to the current episode
        current_episode = int(match_episode.group(1))
        total_actions = int(match_episode.group(2))
        episode_numbers.append(current_episode)
        negative_rewards = 0  # Reset negative rewards count
    
    # Match actions with negative reward
    match_negative_reward = re.search(r'Reward: -\d+\.\d+', line)
    if match_negative_reward:
        negative_rewards += 1

# Add the last episode's data
if current_episode is not None:
    percent_negative_rewards.append((negative_rewards / total_actions) * 100)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(episode_numbers, percent_negative_rewards, marker='o', label="Negative Reward Actions")
plt.title("Percentage of Negative Reward Actions per Episode")
plt.xlabel("Episode Number")
plt.ylabel("Percentage of Negative Actions (%)")
plt.grid(True)
plt.legend()
plt.show()
