import numpy as np  # type: ignore
import random
import pickle
import csv
import os
from collections import defaultdict

class QLearning:
    def __init__(self, env, alpha=0.2, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.05, log_file="metrics.csv"):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.log_file = log_file

        # Simplified action space: throttle, left+throttle, right+throttle
        self.discrete_actions = [
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([-1.0, 1.0, 0.0]),  # Left + throttle
            np.array([1.0, 1.0, 0.0]),   # Right + throttle
        ]
        self.action_space_size = len(self.discrete_actions)

        # Q-table and state visit counter
        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        self.state_visits = defaultdict(int)

        # Initialize log file
        self._init_log()

    def _init_log(self):
        """
        Initialize CSV file with headers if it doesn't exist.
        """
        if not os.path.isfile(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'total_reward', 'epsilon', 'unique_states_seen', 'max_q_value'])

    def _log_metrics(self, episode, total_reward, epsilon, unique_states, max_q):
        """
        Append one row of metrics to CSV.
        """
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, epsilon, unique_states, max_q])

    def discretize_state(self, observation):
        """
        Convert observation to coarse grayscale and downsample more aggressively.
        """
        gray = np.round(observation.mean(axis=2) / 30)   # Compress grayscale more: 0–8 range
        downsampled = gray[::12, ::12]  # 96x96 ⇒ 8x8
        return tuple(downsampled.flatten())

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.Q[state])

    def train(self, num_episodes=500, save_interval=100, verbose=False):
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            total_reward = 0
            episode_over = False
            max_q_value = float('-inf')

            while not episode_over:
                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.discretize_state(next_obs)
                episode_over = terminated or truncated

                # Clip reward to reduce noise
                reward = np.clip(reward, -1.0, 1.0)

                # Q-learning update
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action_index]
                self.Q[state][action_index] += self.alpha * td_error

                # Track maximum Q-value
                max_q_value = max(max_q_value, np.max(self.Q[state]))

                # Bookkeeping
                self.state_visits[state] += 1
                state = next_state
                total_reward += reward

                if verbose:
                    print(f"Step reward: {reward:.2f}")

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Logging
            unique_states = len(self.state_visits)
            self._log_metrics(episode + 1, total_reward, self.epsilon, unique_states, max_q_value)

            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Max Q: {max_q_value:.2f}, Epsilon: {self.epsilon:.3f}")

            if (episode + 1) % save_interval == 0:
                self.save_q_table(f"q_table_ep{episode + 1}.pkl")

        print(f"Training complete. Total unique states seen: {len(self.state_visits)}")
        self.save_q_table("q_table_final.pkl")

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            self.Q = defaultdict(lambda: np.zeros(self.action_space_size), pickle.load(f))
