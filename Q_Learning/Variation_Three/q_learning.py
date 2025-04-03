import numpy as np
import random
import pickle
from collections import defaultdict

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Simplified discrete action space for CarRacing-v3
        self.discrete_actions = [
            np.array([0.0, 1.0, 0.0]),   # Throttle
            np.array([-1.0, 1.0, 0.0]),  # Left + throttle
            np.array([1.0, 1.0, 0.0]),   # Right + throttle
        ]
        self.action_space_size = len(self.discrete_actions)

        # Q-table and visit tracking
        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        self.state_visits = defaultdict(int)

    def discretize_state(self, observation):
        """
        Convert RGB image to a low-resolution, compressed grayscale state.
        Returns a hashable tuple for Q-table.
        """
        gray = np.mean(observation, axis=2)            # Convert to grayscale
        gray = np.round(gray / 20).astype(np.uint8)    # Compress 0–255 → 0–12
        downsampled = gray[::8, ::8]                    # Downsample 96×96 → 12×12
        return tuple(downsampled.flatten())

    def select_action(self, state):
        """
        Epsilon-greedy policy: explore randomly or exploit best-known action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.Q[state]))

    def train(self, num_episodes=500, save_interval=100, verbose=False, seed=None):
        rewards_per_episode = []

        for episode in range(num_episodes):
            observation, info = self.env.reset(seed=seed)
            state = self.discretize_state(observation)
            total_reward = 0
            done = False

            while not done:
                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated
                next_state = self.discretize_state(next_obs)

                # Q-learning update
                best_next = np.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[state][action_index]
                self.Q[state][action_index] += self.alpha * td_error

                # Bookkeeping
                self.state_visits[state] += 1
                state = next_state
                total_reward += reward

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

            # Logging
            if verbose or (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Unique States: {len(self.state_visits)}")

            # Save Q-table
            if (episode + 1) % save_interval == 0:
                self.save_q_table(f"q_table_ep{episode + 1}.pkl")

        print("Training complete.")
        self.save_q_table("q_table_final.pkl")
        return rewards_per_episode

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            q_data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(self.action_space_size), q_data)
