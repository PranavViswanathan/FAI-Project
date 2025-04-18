"""
The actual Q-Learning algo we are implementing
Includes:
State discretization for observation that is imaged based 
We are encouraging the agent to keep on the track and avoid green
Eplison greedy exploration with a steep decay 
Adaptive learning rate based on how many states visited
We save and load the Q table 
"""



import numpy as np  # type: ignore
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.997, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.discrete_actions = [
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([-1.0, 1.0, 0.0]),  # Left + throttle
            np.array([1.0, 1.0, 0.0]),   # Right + throttle
            np.array([0.0, 0.0, 0.8]),   # Brake only
        ]
        self.action_space_size = len(self.discrete_actions)

        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        self.state_visits = defaultdict(int)
        self.rewards = []

    def discretize_state(self, observation):
        gray = np.round(observation.mean(axis=2) / 40)  # Compress to 0–6
        downsampled = gray[::12, ::12]  # 96x96 → 8x8
        return tuple(downsampled.flatten())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.Q[state])

    def is_on_grass(self, observation):
        green_channel = observation[84:, :, 1]  # Bottom portion
        return np.mean(green_channel) > 150

    def train(self, num_episodes=500, save_interval=100, verbose=False):
        self.prev_obs = None

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            self.prev_obs = observation
            state = self.discretize_state(observation)

            total_reward = 0
            episode_over = False
            grass_steps = 0
            total_steps = 0
            speed_sum = 0.0

            while not episode_over:
                # Frame smoothing
                smoothed_obs = (0.7 * observation + 0.3 * self.prev_obs).astype(np.uint8)
                self.prev_obs = smoothed_obs
                state = self.discretize_state(smoothed_obs)

                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                episode_over = terminated or truncated
                total_steps += 1
                speed = info.get("speed", 0.0)
                speed_sum += speed

                # Reward shaping
                if self.is_on_grass(next_obs):
                    grass_steps += 1
                    reward -= min(10, 5 + grass_steps * 0.2)
                else:
                    grass_steps = 0
                    reward += 0.5  # On track bonus

                reward += 0.1 * speed  # Forward motion incentive

                # Penalize sharp steering on road
                if np.abs(action[0]) > 0.5 and grass_steps == 0:
                    reward -= 0.2

                # Clip rewards to stabilize Q-values
                reward = np.clip(reward, -20, 20)

                next_state = self.discretize_state(next_obs)

                # Q-learning update with adaptive alpha
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action_index]
                alpha = max(0.05, 0.5 / (1 + self.state_visits[state]))
                self.Q[state][action_index] += alpha * td_error

                self.state_visits[state] += 1
                state = next_state
                observation = next_obs
                total_reward += reward

                if verbose:
                    print(f"Step reward: {reward:.2f}")

                if grass_steps > 50:
                    print("Too much time on grass, ending episode early.")
                    break

            if episode >= 20:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            avg_speed = speed_sum / max(1, total_steps)
            self.rewards.append(total_reward)

            print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.3f}, Grass steps: {grass_steps}, "
                  f"Avg speed: {avg_speed:.2f}, Unique states: {len(self.state_visits)}")

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

    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.show()
