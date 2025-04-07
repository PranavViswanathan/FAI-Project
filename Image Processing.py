import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque


class ImageProcessing(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, initial_no_op=50):
        super(ImageProcessing, self).__init__(env)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

        self.frames = deque(maxlen=self.stack_frames)

    def preprocess(img):
        img = cv2.resize(img, dsize = (84,84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def reset(self):
        state,info = self.env.reset()

        for _ in range(self.initial_no_op):
            state, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                state, info = self.env.reset()

        state = self.preprocess(state)

        for _ in range(self.stack_frames):
            self.frames.append(state)

        stacked_state = np.stack(self.frames, axis=0)

        return stacked_state, info
    
    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        for _ in range(self.skip_frames):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        state = self.preprocess(state)

        self.frames.append(state)

        stacked_state = np.stack(self.frames, axis=0)

        return stacked_state, total_reward, terminated, truncated, info