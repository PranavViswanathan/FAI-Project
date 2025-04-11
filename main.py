"""
DQN training script for our env

This main script inits and trains our DQ network agent, using:
Preprocessed CarRacing env with (grayscale, stacked frames, frame skipping)
CNN based Q-Network for learning from randomly sampled data from
Exp replay with Epsilon-greedy exploration with steep decay

Tracks rewards per episode and can be visaulized
"""


import gymnasium as gym  
import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2
import os
import string
import random

from ImageProcessing import Observation_processing
from DeepQ import DQN

env = gym.make("CarRacing-v3", render_mode=None, continuous=False)
env = Observation_processing(env)

agent = DQN(stacked_input=(4, 84, 84), num_actions=env.action_space.n)

# returns the path based on greedy "best" actions
def greedy_path(agent):
    greedy_env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')
    greedy_env = Observation_processing(greedy_env)

    frames = []   #list to store the frames
    (current_state, _), done, cumulative_reward = greedy_env.reset(), False, 0    # reset current state, done=False, total reward = 0
    while not done:
        frames.append(greedy_env.render())      #append the rendered frames
        action = agent.act(current_state, training=False) #only return the best qction i.e. argmax(Q(s))
        next_state, reward, terminated, truncated, info = greedy_env.step(action) 
        current_state = next_state
        cumulative_reward += reward
        done = terminated or truncated
    return frames, round(cumulative_reward, 3)     #round up the reward

# for saving the video of best epsiode
def animate(imgs, video_name=None, _return=True):
    if video_name is None:
        video_name = ''.join(random.choice(string.ascii_letters) for _ in range(18)) + '.webm'   #random name to saved video
    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')    # video codec VP90 for .webm video
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #opencv needs RGB, gym returns bgr
        video.write(img)
    video.release()

    if _return:
        from IPython.display import Video
        return Video(video_name)

num_episodes = 2000   # assumed each episode ~ 1000 steps, so max_steps = 2million
max_steps = int(2e6)  # num of actions that can be taken in each episode 
interval = 10000      # review after 10000 steps
episode_rewards = []
history = {'Step': [], 'AvgReturn': []}

total_steps = 0
for episode in range(num_episodes):
    (current_state, _), done = env.reset(), False
    total_reward = 0

    while not done:
        action = agent.act(current_state, training=True)    #epsilon greedy action i.e. if rand<epsilon => random action else argmax(Q(s))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        result = agent.process((current_state, [action], [reward], next_state, [done]))    #store in Experience Replay
        # update state, reward
        current_state = next_state  
        total_reward += reward

        #review every "interval" 
        if agent.total_steps % interval == 0 and agent.total_steps > 0:
            frames, avg_return = greedy_path(agent)    #run the greedy path till now
            history['Step'].append(agent.total_steps) 
            history['AvgReturn'].append(avg_return)   # avg_return = reward in this greedy path

            from IPython.display import clear_output
            clear_output()
            plt.figure(figsize=(8, 5))
            plt.plot(history['Step'], history['AvgReturn'], 'r-')
            plt.xlabel('Step', fontsize=16)
            plt.ylabel('AvgReturn', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')
            plt.show()

            # save the values
            torch.save(agent.network.state_dict(), 'dqn.pt')

            #video
            animate(frames, "trial1.webm")

        #additional condition to end training
        if agent.total_steps > max_steps:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Steps so far: {agent.total_steps}")


#starting 4 grayscale images

# state, _ = env.reset()
# print("The shape of an observation: ", state.shape)

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for i in range(4):
#     axes[i].imshow(state[i], cmap='gray')
#     axes[i].axis('off')
# plt.show()