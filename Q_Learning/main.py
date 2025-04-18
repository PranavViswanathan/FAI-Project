"""
This is the seventh variation (gen7) of the Q learning table
The variations can be examined on Pranav's feature branch
Each variations added incremental design changes
Among of which in generation 7 we implemented bucketing
Which greatly converted our results to converge to consistent positive values

The code below is the main entry point to our code
It's responsible for running the environment with our specifc parameters
It implements the q_learning algo we implemented
Runs with moderate success
Ultimately it falls short of creating a Q table that could be considered succesful
"""


import gymnasium as gym
from qlearning import QLearning
import matplotlib.pyplot as plt
import csv
import os

# === Configuration ===
num_episodes = 2000
save_interval = 500
window_size = 10
csv_filename = "Q_Learning/metrics.csv"
TRACK_SEED = 42  # Fixed seed for same track every episode

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Initialize the Q-learning agent
agent = QLearning(env)

# Metrics
rewards = []
avg_rewards = []
epsilons = []
unique_states = []

# Create CSV file and write header
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Reward", "Avg10", "Epsilon", "UniqueStates"])

    for episode in range(num_episodes):
        observation, info = env.reset(seed=TRACK_SEED)  # Fixed seed
        state = agent.discretize_state(observation)
        total_reward = 0
        done = False

        while not done:
            action_index = agent.select_action(state)
            action = agent.discrete_actions[action_index]

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)

            # Q-learning update
            best_next = max(agent.Q[next_state])
            td_target = reward + agent.gamma * best_next
            td_error = td_target - agent.Q[state][action_index]
            agent.Q[state][action_index] += agent.alpha * td_error

            # Update state
            agent.state_visits[state] += 1
            total_reward += reward
            state = next_state
            done = terminated or truncated

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Metrics collection
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        unique_states.append(len(agent.state_visits))
        if episode >= window_size:
            avg = sum(rewards[-window_size:]) / window_size
            avg_rewards.append(avg)
        else:
            avg_rewards.append(total_reward)

        # Save metrics to CSV
        writer.writerow([
            episode + 1,
            total_reward,
            avg_rewards[-1],
            agent.epsilon,
            unique_states[-1]
        ])

        # Console output
        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Avg10: {avg_rewards[-1]:.2f}, "
              f"Epsilon: {agent.epsilon:.3f}, Unique states: {unique_states[-1]}")

        # Periodic Q-table saving
        if (episode + 1) % save_interval == 0:
            q_filename = f"q_table_ep{episode+1}.pkl"
            agent.save_q_table(q_filename)
            print(f"Saved Q-table to {q_filename}")

# Final save
agent.save_q_table("q_table_final.pkl")
env.close()

# === Plotting ===
plt.figure(figsize=(12, 8))

# Reward Plot
plt.subplot(3, 1, 1)
plt.plot(rewards, label='Total Reward')
plt.plot(avg_rewards, label='10-Episode Moving Avg', linestyle='--')
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

# Epsilon Plot
plt.subplot(3, 1, 2)
plt.plot(epsilons)
plt.title('Epsilon Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

# Unique States Plot
plt.subplot(3, 1, 3)
plt.plot(unique_states)
plt.title('Unique States Seen Over Time')
plt.xlabel('Episode')
plt.ylabel('Unique States')

plt.tight_layout()
plt.show()
