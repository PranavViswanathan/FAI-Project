import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNCNN(nn.Module):
    def __init__(self, action_size):
        super(DQNCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # Input: grayscale (1 channel)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class DQLearning:
    def __init__(self, env, alpha=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        self.discrete_actions = [
            np.array([-1.0, 0.0, 0.0]),  # Hard left
            np.array([1.0, 0.0, 0.0]),   # Hard right
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([0.0, 0.0, 1.0]),   # Full brake
            np.array([0.0, 0.0, 0.0]),   # Do nothing
        ]
        self.action_size = len(self.discrete_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNCNN(self.action_size).to(self.device)
        self.target_net = DQNCNN(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.update_target()

    def preprocess(self, obs):
        gray = np.mean(obs, axis=2).astype(np.uint8)  # Convert to grayscale
        resized = gray[::4, ::4]  # Downsample
        normalized = resized / 255.0
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=500, target_update_freq=10):
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = self.preprocess(obs)
            total_reward = 0
            done = False

            while not done:
                action_index = self.select_action(state)
                action = self.discrete_actions[action_index]
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.preprocess(next_obs)

                done = terminated or truncated
                self.store_transition(state, action_index, reward, next_state, done)
                self.train_step()

                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % target_update_freq == 0:
                self.update_target()

            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")
