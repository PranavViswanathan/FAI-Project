import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import trange


# Neural network for Q-value approximation
class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0  # Normalize
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)


# Preprocess RGB observation to 84x84 grayscale
def preprocess(observation):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    return transform(observation).squeeze(0).numpy()


# Discretized action space
ACTION_SPACE = [
    np.array([0.0, 1.0, 0.0]),   # Throttle
    np.array([-1.0, 1.0, 0.0]),  # Left + throttle
    np.array([1.0, 1.0, 0.0]),   # Right + throttle
    np.array([0.0, 0.0, 0.8]),   # Brake
]


# DQN Agent Class
class DQNAgent:
    def __init__(self, state_shape, action_space_size, device):
        self.device = device
        self.action_space_size = action_space_size
        self.policy_net = DQNCNN(state_shape, action_space_size).to(device)
        self.target_net = DQNCNN(state_shape, action_space_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_steps = 1000
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.unsqueeze(1).to(self.device)
        next_states = next_states.unsqueeze(1).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Main training loop
def train_dqn(num_episodes=300):
    env = gym.make("CarRacing-v2", continuous=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent((1, 84, 84), len(ACTION_SPACE), device)

    rewards = []
    for episode in trange(num_episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            action = ACTION_SPACE[action_idx]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)

            agent.push(state, action_idx, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    plt.plot(rewards)
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_dqn()
