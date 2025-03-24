import numpy as np
import random
from collections import defaultdict

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01): #change the decay rate to higher for more exploration
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        # Define a set of discrete actions
        self.discrete_actions = [
            np.array([-1.0, 0.0, 0.0]),  # Hard left
            np.array([1.0, 0.0, 0.0]),   # Hard right
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([0.0, 0.0, 1.0]),   # Full brake
            np.array([0.0, 0.0, 0.0]),   # Do nothing
        ]
        self.action_space_size = len(self.discrete_actions)

        # Initialize the Q-table
        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))


    #TODO
    #needs explanation what it does, might be computer vision 
    def discretize_state(self, observation):
        """
        Discretize the observation (state) into a simpler form.
        For example, reduce the image to a smaller resolution and convert to grayscale.
        """
        observation = observation.mean(axis=2)  # Convert to grayscale by averaging RGB channels
        observation = observation[::10, ::10]   # Downsample to 10x10
        return tuple(observation.flatten())     # Flatten and convert to a tuple for hashing


    def select_action(self, state):
        """
        Select an action using the epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore: random discrete action
        else:
            return np.argmax(self.Q[state])  # Exploit: best action from Q-table


    #TODO
    #add what is the total reward/result
    def train(self, num_episodes=10):
        """
        Train the Q-Learning agent.
        """
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            episode_over = False
            total_reward = 0

            while not episode_over:
                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]  # Map index to discrete action
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.discretize_state(next_observation)
                episode_over = terminated or truncated

                # Update Q-value using the Q-Learning formula
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action_index]
                self.Q[state][action_index] += self.alpha * td_error

                # Update state and total reward
                state = next_state
                total_reward += reward

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

        # Save the Q-table
        self.save_q_table("q_table.pkl")

    
    def save_q_table(self, filename):
        """
        Save the Q-table to a file.
        """
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, filename):
        """
        Load the Q-table from a file.
        """
        import pickle
        with open(filename, "rb") as f:
            self.Q = defaultdict(lambda: np.zeros(self.action_space_size), pickle.load(f))