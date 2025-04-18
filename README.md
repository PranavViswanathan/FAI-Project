# NeuroPilot - A RL based race car simulator

Our project entails taking a 2D car racing basic environment from Gymnasium and investigating the performance of two RL algorithms; tabular Q-learning and DQN. The objective is to train autonomous agents capable of navigating tracks with varying complexities, by learning optimal driving, pathfinding and general policies through trial and error. We first implement Q-learning using discretized states spaces then extend to DQN using convolutional neural networks to handle high dimensional visual input. 

## Features

- **Gymnasium CarRacing-v3 Environment**  
  A 2D continuous-control racing environment, now discretized and optimized for reinforcement learning.
  
- **Tabular Q-Learning Agent**  
  Implements a classical Q-learning agent with state discretization and epsilon-greedy exploration.

- **DQN Agent**  
  Deep Q-Network that uses a CNN to process pixel observations, with experience replay and target network updates.

- **Training Analytics**  
  Plotting and logging of training reward, episode length, and exploration rate.

- **Model Saving & Evaluation**  
  Q-tables and DQN model checkpoints can be saved and reloaded for evaluation or continued training.

##  Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/PranavViswanathan/FAI-Project.git
   cd FAI-Project
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install extra packages for rendering**
   ```bash
   sudo apt-get install ffmpeg xvfb
   ```


## Project Structure
```
├── DQN
│   ├── DeepQ.py
│   ├── ImageProcessing.py
│   ├── checkpoints
│   │   ├── checkpoint_ep1000.pth
│   │   ├── checkpoint_ep1500.pth
│   │   └── checkpoint_ep500.pth
│   ├── main.py
│   ├── plots
│   │   ├── dqn_carracing.pth
│   │   ├── episode_rewards.png
│   │   ├── epsilon_decay.png
│   │   ├── training_metrics.pkl
│   │   └── unique_states.png
│   └── replay.py
├── FAI final prez.pdf
├── Prev_Codebase
│   ├── main.py
│   ├── q_learning.py
│   ├── q_table.pkl
│   └── test_policy.py
├── Q_Learning
│   ├── main.py
│   ├── metrics.csv
│   ├── qlearning.py
│   ├── requirements.txt
│   └── test_policy.py
├── README.md
├── requirements.txt
└── resources
    └── test.md

```
## Training

### 1. Q-Learning
Run the Q-learning agent:
```bash
python Q_Learning/main.py
```

Adjust hyperparameters such as learning rate (`alpha`), discount factor (`gamma`), or epsilon decay inside the script.

### 2. DQN
Run the DQN training:
```bash
python DQN/main.py
```
### 3. Visulaizing the results
To visulize the model run the test policy file in each folder, for example:
```bash
python Q_Learning/test_policy.py

```
