"""
DQN training script for our env

This main script inits and trains our DQ network agent, using:
Preprocessed CarRacing env with (grayscale, stacked frames, frame skipping)
CNN based Q-Network for learning from visual input 
Exp replay //TODO
EPsilon-greedy exploration with steep decay

Tracks rewards per episode and can be visaulized
"""



import gymnasium as gym #type:ignore
import matplotlib.pyplot as plt
import numpy as np
from ImageProcessing import Observation_processing
from DeepQ import DQN

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode=None, continuous=False)  #change render_mode to human if we want to visualize
env = Observation_processing(env)

max_steps = int(2e4)

agent = DQN(stacked_input = (4, 84, 84), num_actions = env.action_space.n)

## TRAINING LOOP
num_episodes = 1000
episode_rewards = []

for episode in range(num_episodes):
    (current_state, _), done = env.reset(), False
    total_reward = 0

    while not done:
        action = agent.act(current_state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.process((current_state, [action], [reward], next_state, [done]))
        current_state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")


#starting 4 grayscale images

# state, _ = env.reset()
# print("The shape of an observation: ", state.shape)

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for i in range(4):
#     axes[i].imshow(state[i], cmap='gray')
#     axes[i].axis('off')
# plt.show()