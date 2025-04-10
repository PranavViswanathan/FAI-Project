import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque


class Observation_processing(gym.Wrapper):
    def __init__(self, env, repeat_action=3, stack_frames=4, do_nothing_frames=50):
        super(Observation_processing, self).__init__(env)
        self.do_nothing_frames = do_nothing_frames
        self.repeat_action = repeat_action #same action for these frames => easy computation
        self.stack_frames = stack_frames #predicting motion of car

        self.frames = deque(maxlen=self.stack_frames)  #refresh the frames

    def rgb_to_grayscale(self, img):
        img = cv2.resize(img, dsize = (84,84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    # reset episode => do nothing -> grayscale -> stack
    def reset(self):
        state,info = self.env.reset()

        # for this zoom in phase do nothing
        for _ in range(self.do_nothing_frames):
            state, _, terminated, truncated, info = self.env.step(0)
            #additional termination condition to avoid bad terminal state
            if terminated or truncated:
                state, info = self.env.reset()

        state = self.rgb_to_grayscale(state)

        #deque 4 frames
        for _ in range(self.stack_frames):
            self.frames.append(state)

        # stack frames
        stacked_state = np.stack(self.frames, axis=0)

        return stacked_state, info # [stack_frames=4, 84, 84]
    
    #take an action
    def step(self, action):
        total_reward = 0 #initializaed
        terminated = False
        truncated = False

        #repeat action for repeat_action
        for _ in range(self.repeat_action):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        state = self.rgb_to_grayscale(state)
        self.frames.append(state) #store new frames; (t-2, t-1, t, t+1)
        stacked_state = np.stack(self.frames, axis=0) 

        return stacked_state, total_reward, terminated, truncated, info