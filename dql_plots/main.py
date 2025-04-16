"""
DQN training script for our env

This main script inits and trains our DQ network agent, using:
Preprocessed CarRacing env with (grayscale, stacked frames, frame skipping)
CNN based Q-Network for learning from visual input 
Exp replay //TODO
EPsilon-greedy exploration with steep decay

Tracks rewards per episode and can be visaulized
"""



import gymnasium as gym #type:ignore
import matplotlib.pyplot as plt
import numpy as np
from ImageProcessing import Observation_processing
from DeepQ import DQN
from PIL import Image
import imageio
import os
from datetime import datetime

# Create directories for saving outputs
os.makedirs('training_plots', exist_ok=True)
os.makedirs('training_gifs', exist_ok=True)

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)  #change render_mode to human if we want to visualize
env = Observation_processing(env)

max_steps = int(2e4)

agent = DQN(stacked_input = (4, 84, 84), num_actions = env.action_space.n)

## TRAINING LOOP
num_episodes = 1000
episode_rewards = []

moving_avg_rewards = []
window_size = 20  # for moving average

# For recording GIFs
record_every = 50  # record a GIF every N episodes
gif_frames = []

for episode in range(num_episodes):
    (current_state, _), done = env.reset(), False
    total_reward = 0

    # For GIF recording
    if episode % record_every == 0:
        episode_frames = []

    while not done:
        action = agent.act(current_state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record frame if this is a recording episode
        if episode % record_every == 0:
            frame = env.env.render()  # Get the RGB frame
            episode_frames.append(frame)

        agent.process((current_state, [action], [reward], next_state, [done]))
        current_state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Calculate moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
    
    # Save GIF for this episode if it's a recording episode
    if episode % record_every == 0 and episode_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f'training_gifs/episode_{episode}_{timestamp}.gif'
        
        # Resize frames to make GIF smaller (optional)
        resized_frames = [Image.fromarray(frame).resize((400, 300)) for frame in episode_frames]
        
        # Save as GIF
        imageio.mimsave(gif_path, resized_frames, duration=0.1)
        print(f"Saved GIF for episode {episode} at {gif_path}")
    
    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Plot and save training progress periodically
    if episode % 10 == 0 or episode == num_episodes - 1:
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, label='Episode Reward', alpha=0.3)
        if moving_avg_rewards:
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg_rewards, label=f'{window_size}-episode MA', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot histogram of recent rewards
        plt.subplot(1, 2, 2)
        recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]
        plt.hist(recent_rewards, bins=20, edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Recent Reward Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f'training_plots/training_progress_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved training plot at {plot_path}")

#starting 4 grayscale images

# state, _ = env.reset()
# print("The shape of an observation: ", state.shape)

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for i in range(4):
#     axes[i].imshow(state[i], cmap='gray')
#     axes[i].axis('off')
# plt.show()