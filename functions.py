import numpy as np

class Q_Learning:
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds, actions):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.actions = actions
        self.actionNumber = len(actions)
        self.sumRewardsEpisode = []

        # Initialize Q-table for 3D state space (offset, angle, speed) and action dimension
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
            numberOfBins[0], numberOfBins[1], numberOfBins[2], self.actionNumber))

    def returnIndexState(self, state):
        offset_bins = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        angle_bins = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        speed_bins = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])

        i_offset = np.clip(np.digitize(state[0], offset_bins) - 1, 0, self.numberOfBins[0] - 1)
        i_angle = np.clip(np.digitize(state[1], angle_bins) - 1, 0, self.numberOfBins[1] - 1)
        i_speed = np.clip(np.digitize(state[2], speed_bins) - 1, 0, self.numberOfBins[2] - 1)

        return (i_offset, i_angle, i_speed)

    def selectAction(self, state, episodeIndex):
        if episodeIndex < 500:
            return np.random.choice(self.actionNumber)

        if episodeIndex > 7000:
            self.epsilon *= 0.999

        if np.random.random() < self.epsilon:
            return np.random.choice(self.actionNumber)
        else:
            state_idx = self.returnIndexState(state)
            best_actions = np.where(self.Qmatrix[state_idx] == np.max(self.Qmatrix[state_idx]))[0]
            return np.random.choice(best_actions)

    def simulateEpisodes(self):
        for episodeIndex in range(self.numberEpisodes):
            rewardsEpisode = []

            self.env.reset()
            stateS = self.env.get_state()

            print(f"Simulating episode {episodeIndex}")
            done = False

            while not done:
                stateS_idx = self.returnIndexState(stateS)
                action_idx = self.selectAction(stateS, episodeIndex)
                action = self.actions[action_idx]

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                rewardsEpisode.append(reward)

                stateSprime = self.env.get_state()
                stateSprime_idx = self.returnIndexState(stateSprime)

                QmaxPrime = np.max(self.Qmatrix[stateSprime_idx])

                if not done:
                    error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateS_idx + (action_idx,)]
                else:
                    error = reward - self.Qmatrix[stateS_idx + (action_idx,)]

                self.Qmatrix[stateS_idx + (action_idx,)] += self.alpha * error
                stateS = stateSprime

            episode_reward_sum = np.sum(rewardsEpisode)
            print(f"Episode {episodeIndex} sum of rewards: {episode_reward_sum}")
            self.sumRewardsEpisode.append(episode_reward_sum)
