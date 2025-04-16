import gymnasium as gym
import torch
import numpy as np
from ImageProcessing import Observation_processing
from DeepQ import DQN  

env = gym.make("CarRacing-v3", render_mode="human", continuous=False)
env = Observation_processing(env)

agent = DQN(stacked_input=(4, 84, 84), num_actions=env.action_space.n)
agent.network.load_state_dict(torch.load("DQN/checkpoints/checkpoint_ep1000.pth"))
agent.network.eval()
agent.epsilon = 0.0  

(current_state, _), done = env.reset(), False
total_reward = 0

while not done:
    action = agent.act(current_state, training=False)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    current_state = next_state
    total_reward += reward

print(f"Finished episode | Total Reward: {total_reward:.2f}")
