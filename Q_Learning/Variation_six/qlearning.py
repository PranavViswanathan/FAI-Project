import numpy as np  # type: ignore
import random
import pickle
from collections import defaultdict, deque


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Expanded action space
        self.discrete_actions = [
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([-1.0, 1.0, 0.0]),  # Left + throttle
            np.array([1.0, 1.0, 0.0]),   # Right + throttle
            np.array([0.0, 0.0, 0.8]),   # Brake
        ]
        self.action_space_size = len(self.discrete_actions)

        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        self.state_visits = defaultdict(int)

    def discretize_state(self, observation):
        gray = np.round(observation.mean(axis=2) / 20)
        downsampled = gray[::8, ::8]
        return tuple(downsampled.flatten())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.Q[state])

    def is_on_grass(self, observation):
        green_channel = observation[84:, :, 1]
        return np.mean(green_channel) > 150

    def train(self, num_episodes=500, save_interval=100, verbose=False):
        loop_check_window = 20
        loop_threshold = 15

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            total_reward = 0
            episode_over = False
            grass_steps = 0

            recent_states = deque(maxlen=loop_check_window)
            loop_counter = 0

            while not episode_over:
                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Grass penalty / track bonus
                if self.is_on_grass(next_obs):
                    grass_steps += 1
                    reward -= (10 + grass_steps * 0.5)
                else:
                    grass_steps = 0
                    reward += 1

                next_state = self.discretize_state(next_obs)

                # Loop detection
                recent_states.append(next_state)
                if recent_states.count(next_state) > loop_threshold:
                    print("🌀 Agent is stuck — ending episode early.")
                    episode_over = True

                episode_over = episode_over or terminated or truncated

                # Q-update
                best_next = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next]
                td_error = td_target - self.Q[state][action_index]
                self.Q[state][action_index] += self.alpha * td_error

                self.state_visits[state] += 1
                state = next_state
                total_reward += reward

                if verbose:
                    print(f"Step reward: {reward:.2f}")

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

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
