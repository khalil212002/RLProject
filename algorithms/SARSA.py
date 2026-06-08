import numpy as np


class SARSA:
    def __init__(self, stateCount, actionCount, stepSize, discount, epsilon):
        self.stateCount = stateCount
        self.actionCount = actionCount
        self.stepSize = stepSize
        self.discount = discount
        self.epsilon = epsilon

        self.qMatrix = np.zeros((stateCount, actionCount))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actionCount)
        return np.argmax(self.qMatrix[state, :])

    def _TDError(self, state, action, reward, next_state, next_action, terminated):
        if terminated:
            return reward - self.qMatrix[state, action]
        return (
            reward
            + self.discount * self.qMatrix[next_state, next_action]
            - self.qMatrix[state, action]
        )

    def update(self, state, action, reward, next_state, next_action, terminated):
        error = self._TDError(
            state, action, reward, next_state, next_action, terminated
        )
        self.qMatrix[state, action] += self.stepSize * error
