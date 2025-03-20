#driverCode.py

import numpy as np
import matplotlib.pyplot as plt
from functions import Q_Learning
from main import CustomCarRacing

# Define discrete actions for the car
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # accelerate forward
    np.array([-0.5, 1.0, 0.0]),  # steer left & accelerate
    np.array([0.5, 1.0, 0.0]),   # steer right & accelerate
    np.array([0.0, 0.0, 0.8]),   # brake
    np.array([0.0, 0.0, 0.0])    # coast
]

# Initialize the custom car racing environment
env = CustomCarRacing(track_complexity=1.5, track_width=0.8, num_forks=2)

# Define state discretization bounds (offset, angle, speed)
lowerBounds = np.array([-1.0, -3.14, 0.0])
upperBounds = np.array([1.0, 3.14, 100.0])
numberOfBins = [10, 10, 10]

# Define Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.2
numberEpisodes = 5000  # You can adjust for longer training

# Create Q-learning object
Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds, ACTIONS)

Q1.simulateEpisodes()

env.close()  # <- add this line

# Plot reward convergence over episodes
plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.title('Q-Learning Reward Convergence')
plt.savefig('convergence.png')
plt.show()

np.save('qmatrix.npy', Q1.Qmatrix)
print("Training complete. Q-matrix saved.")