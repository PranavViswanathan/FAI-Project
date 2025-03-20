# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# from main import CustomCarRacing

# print("Initializing Custom Car Racing Environment...")
# env = CustomCarRacing(track_complexity=1.5, track_width=0.8, num_forks=2)

# print("Resetting environment...")
# obs, info = env.reset()
# print("Environment reset complete!")

# done = False

# print("Extracting track points...")
# track_points = np.array(env.track)
# print(f"Total track points: {len(track_points)}")

# # Check if track points are empty
# if len(track_points) == 0:
#     print("Warning: No track points found! Check track generation logic.")

# print("Plotting the track...")
# plt.figure(figsize=(8, 8))

# # Plot the main track
# plt.plot(track_points[:, 0], track_points[:, 1], 'g', label="Main Track")

# # Mark the individual track points
# plt.scatter(track_points[:, 0], track_points[:, 1], c='blue', s=10)  

# plt.legend()
# plt.title("Custom Car Racing Track")
# print("Displaying the track plot...")
# plt.show()

# print("Closing the environment...")
# env.close()
# print("Environment closed successfully!")


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from main import CustomCarRacing

# print("[INFO] Creating CustomCarRacing environment...")
# env = CustomCarRacing(track_complexity=1.5, track_width=0.8, num_forks=2)

# print("[INFO] Resetting environment...")
# obs, info = env.reset()
# print("[INFO] Environment reset successful!")

# env.close()
# print("[INFO] Closed environment.")

env = gym.make("CarRacing", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
