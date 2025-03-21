import gymnasium as gym
from q_learning import QLearning

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode=None)  

# Initialize the Q-Learning agent
q_learning_agent = QLearning(env)

# Train the agent
q_learning_agent.train(num_episodes=10)

# Close the environment
env.close()