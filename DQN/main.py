"""
DQN training script for our env

This main script inits and trains our DQ network agent, using:
Preprocessed CarRacing env with (grayscale, stacked frames, frame skipping)
CNN based Q-Network for learning from randomly sampled data from
Exp replay with Epsilon-greedy exploration with steep decay

Tracks rewards per episode, epsilon decay, and unique states seen
"""

import gymnasium as gym  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from ImageProcessing import Observation_processing
from DeepQ import DQN

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode=None, continuous=False)
env = Observation_processing(env)

agent = DQN(stacked_input=(4, 84, 84), num_actions=env.action_space.n)

# === Logging metrics ===
episode_rewards = []
epsilons = []
unique_states_seen = set()
unique_states_per_episode = []

# === Training ===
num_episodes = 1_500

for episode in range(num_episodes):
    (current_state, _), done = env.reset(), False
    total_reward = 0

    while not done:
        action = agent.act(current_state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Flatten and hash the state for uniqueness tracking
        unique_states_seen.add(next_state.tobytes())

        agent.process((current_state, [action], [reward], next_state, [done]))
        current_state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    epsilons.append(agent.epsilon)
    unique_states_per_episode.append(len(unique_states_seen))

    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Unique States: {len(unique_states_seen)}")

    # Save progress every 500 episodes
    if (episode + 1) % 500 == 0:
        torch.save(agent.network.state_dict(), f"DQN/checkpoints/checkpoint_ep{episode+1}.pth")
        with open("DQN/plots/training_metrics.pkl", "wb") as f:
            pickle.dump({
                "rewards": episode_rewards,
                "epsilons": epsilons,
                "unique_states_count": unique_states_per_episode
            }, f)
        print(f"[Checkpoint] Saved at Episode {episode+1}")
    

# === Save model ===
torch.save(agent.network.state_dict(), "DQN/plots/dqn_carracing.pth")

# === Save training data for reuse ===
with open("DQN/plots/training_metrics.pkl", "wb") as f:
    pickle.dump({
        "rewards": episode_rewards,
        "epsilons": epsilons,
        "unique_states_count": unique_states_per_episode
    }, f)

# === Plot graphs ===
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards")
plt.savefig("DQN/plots/episode_rewards.png")

plt.figure()
plt.plot(epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")
plt.savefig("DQN/plots/epsilon_decay.png")

plt.figure()
plt.plot(range(len(episode_rewards)), unique_states_per_episode)
plt.xlabel("Episode")
plt.ylabel("Unique States Seen")
plt.title("Unique States Over Time")
plt.savefig("DQN/plots/unique_states.png")