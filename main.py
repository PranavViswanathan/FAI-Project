import gymnasium as gym #type:ignore
from q_learning import QLearning

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode=None)  #change render_mode to human if we want to visualize

# Initialize the Q-Learning agent
q_learning_agent = QLearning(env)

# Train the agent
q_learning_agent.train(num_episodes=10)

# Close the environment
env.close()