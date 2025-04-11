"""
This is the DQN implementation with Convolutional Q-Network 

In this we have:
QNetwork: a Convolusional Network to estimate Q values from image input
DQN: the Deep Q-Learning agent class that handles action slection, training, target network updates and epsilon decay

The important bits are:
Convolutional layers that we use for visual input processing
Epsilon greedy exploration just like in Q-Learning
Experience replay: In this case 5 arrays which store the transitions and get overwritten with the new ones, so overall behaving like a Python queue. // 
RMSProp optimizer: Uses running avg of squared gradients to scale updates and adjusts the laerning rate of parameters instead of constant parameters. //
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from collections import namedtuple, deque

#Neural network we use to get the Q values from visual input(stacked frames)
class QNetwork(nn.Module):
    def __init__(self, stacked_input, num_actions, activation=F.relu):
        super(QNetwork, self).__init__()
        #convolution 1st layer
        self.layer1 = nn.Conv2d(stacked_input, 16, kernel_size=8, stride=4)  # [batch_size, 4, 84, 84] -> [batch_size, 16, 20, 20]   Filters = 16
        
        #convolution 2nd layer
        self.layer2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [batch_size, 32, 9, 9] input from 1st layer = 16, filters =32
        
        #Ouput size after 2nd layer
        self.flatten_img = 32 * 9 * 9  # layer 2 gives out 32 filtered, 9x9 sized batch_size times -> [batch_size, 2592]
        
        #First fully connected layer 
        self.fully_connected_layer1 = nn.Linear(self.flatten_img, 256)   #flattened img passed to 256 neurons
        self.fully_connect_layer2 = nn.Linear(256, num_actions)  #Q values  [batch_size, 3]
        
        self.activation = activation

    def forward(self, input_img):
        #apply conv layers with ReLU
        input_img = F.relu(self.layer1(input_img))
        input_img = F.relu(self.layer2(input_img))
        input_img = input_img.view((-1, self.flatten_img))
        input_img = self.activation(self.fully_connected_layer1(input_img))
        input_img = self.fully_connect_layer2(input_img)
        return input_img

TrainingSample = namedtuple('TraniningSample', ('state', 'action', 'reward', 'next_state', 'terminated'))   #datastructure to access the tuple using labels, 'Transition' in Pytorch

#Experience Replay => stores the transitions (Training Sample); randomly sample a batch //
class ExperienceReplay:
    def __init__(self, stacked_input, num_actions, capacity=int(1e5)):
        self.capacity = capacity
        self.sample_idx = 0
        self.samples_stored_till_now = 0 

        #initiliazing
        self.state = np.zeros((capacity, *stacked_input), dtype=np.float32)
        self.action = np.zeros((capacity, *num_actions), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, *stacked_input), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)

    #step in environment
    def push(self, state, action, reward, next_state, terminated):
        self.state[self.sample_idx] = state
        self.action[self.sample_idx] = action
        self.reward[self.sample_idx] = reward
        self.next_state[self.sample_idx] = next_state
        self.terminated[self.sample_idx] = terminated

        self.sample_idx = (self.sample_idx + 1) % self.capacity
        self.samples_stored_till_now = min(self.samples_stored_till_now + 1, self.capacity)   # rewrite the old memory idx

    #random batch
    def sample(self, batch_size):
        idx = np.random.randint(0, self.samples_stored_till_now, batch_size)
        batch = TrainingSample(
            state=torch.FloatTensor(self.state[idx]),    
            action=torch.LongTensor(self.action[idx]),
            reward=torch.FloatTensor(self.reward[idx]),
            next_state=torch.FloatTensor(self.next_state[idx]),
            terminated=torch.FloatTensor(self.terminated[idx]),
        )
        return batch

    #how much samples are stored
    def __len__(self):
        return self.samples_stored_till_now

#Deep Q-learning Network   
class DQN:
    def __init__(
        self,
        stacked_input,
        num_actions,
        alpha=0.0001,   #learning rate
        epsilon=1.0,     # Epsilon for Epsilon Greedy Algo
        minimum_epsilon=0.1,  # lower bound of Epsilon
        discount_factor=0.99, # discount factor
        batch_size=32,   #batch size input to neural network
        warmup_steps=10000,   #steps where the agent collects experience but doesn’t learn, improves randomness in replay buffer data
        ExperienceReplay_memory=int(1e5),
        target_update_interval=5000,
    ):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon  # storing decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        #Q Network; input => stacked frames, action; output => QValues
        self.network = QNetwork(stacked_input[0], num_actions)   

        #Target Network
        self.target_network = QNetwork(stacked_input[0], num_actions)

        #update the weights in Target Network
        self.target_network.load_state_dict(self.network.state_dict()) 

        #optimizer; reference => Deepmind DQN Paper
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), alpha) 

        self.buffer = ExperienceReplay(stacked_input, (1, ), ExperienceReplay_memory) #initialized Experience Replay
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - minimum_epsilon) / 3e5  #Epsilon Decay
        #self.decay_rate = 0.9990  # Tune this to control the curve

    
    #Epsilon Greedy
    @torch.no_grad()
    def act(self, input_img, training=True):
        self.network.eval() if not training else self.network.train()
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            action = np.random.randint(0, self.num_actions)   
        else:
            input_img = torch.from_numpy(input_img).float().unsqueeze(0)
            q = self.network(input_img)
            action = torch.argmax(q).item()
        return action
    #Perform a training step 
    def learn(self):
        current_state, action, reward, next_state, terminated = self.buffer.sample(self.batch_size) # random batch of past transitions from replay buffer and move them to GPU.
        
        # Q(s', a)
        next_q = self.target_network(next_state).detach()
        #Get target Q-values from network
        target_q = reward + (1. - terminated) * self.discount_factor * next_q.max(dim=1, keepdim=True).values   # target = immediate reward + gamma * Q(s', a)
        #Loss
        loss = F.mse_loss(self.network(current_state).gather(1, action.long()), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result
    #Proccess a single transition and update networks 
    def process(self, transition):
        result = {}
        self.total_steps += 1

        #sotre transition 
        self.buffer.push(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        # update weights
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        #decay epsilon
        #self.epsilon -= self.epsilon_decay #linear decay
        self.epsilon = max(self.minimum_epsilon, self.epsilon - self.epsilon_decay)

        return result
