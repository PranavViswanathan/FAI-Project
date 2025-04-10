"""
Testing the model by checking how well our model is performing 
This was used for fixing the exploration instead of explotation 
This was used as a sanity check
"""


import gymnasium as gym # type: ignore
from Variation_One.q_learning import QLearning
import numpy as np   # type: ignore

# Create the Car Racing environment with rendering enabled during testing
env = gym.make("CarRacing-v3", render_mode="human")  # Enable rendering

# Initialize the Q-Learning agent
q_learning_agent = QLearning(env)  

# Load the Q-table
q_learning_agent.load_q_table("Variation_Four/q_table_ep1000.pkl")

# Test the learned policy
observation, info = env.reset()
state = q_learning_agent.discretize_state(observation)
episode_over = False
total_reward = 0

#Run the episodes using the learned policy ( no exploration, always best action)
while not episode_over:
    action_index = np.argmax(q_learning_agent.Q[state])  # Always choose the best action
    action = q_learning_agent.discrete_actions[action_index]  # Map index to discrete action
    observation, reward, terminated, truncated, info = env.step(action)
    state = q_learning_agent.discretize_state(observation)
    episode_over = terminated or truncated
    total_reward += reward
    env.render()

#Print all rewards
print(f"Total Reward: {total_reward}")
env.close()