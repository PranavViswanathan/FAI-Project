import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("Variation_Five/metrics.csv")

# Create subplots
plt.figure(figsize=(14, 10))

# Plot 1: Reward per Episode
plt.subplot(2, 2, 1)
plt.plot(df['Episode'], df['Reward'], label='Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.grid(True)

# Plot 2: Average of Last 10 Rewards
plt.subplot(2, 2, 2)
plt.plot(df['Episode'], df['Avg10'], label='Avg10', color='orange')
plt.xlabel("Episode")
plt.ylabel("Average Reward (Last 10)")
plt.title("Moving Average of Reward (Last 10)")
plt.grid(True)

# Plot 3: Epsilon Decay
plt.subplot(2, 2, 3)
plt.plot(df['Episode'], df['Epsilon'], label='Epsilon', color='green')
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Time")
plt.grid(True)

# Plot 4: Unique States Seen
plt.subplot(2, 2, 4)
plt.plot(df['Episode'], df['UniqueStates'], label='UniqueStates', color='purple')
plt.xlabel("Episode")
plt.ylabel("Unique States")
plt.title("Unique States Seen Over Time")
plt.grid(True)

# Adjust layout and show
plt.tight_layout()
plt.show()
