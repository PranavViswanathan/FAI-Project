import gymnasium as gym #type:ignore
import matplotlib.pyplot as plt
from ImageProcessing import Observation_processing
from DeepQ import DQN

# Create the Car Racing environment
env = gym.make("CarRacing-v3", render_mode=None, continuous=False)  #change render_mode to human if we want to visualize
env = Observation_processing(env)

#starting 4 grayscale images
state, _ = env.reset()
print("The shape of an observation: ", state.shape)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(4):
    axes[i].imshow(state[i], cmap='gray')
    axes[i].axis('off')
plt.show()