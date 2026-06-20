from algorithms.SARSA import SARSA
import numpy as np


class SARSAGen(SARSA):
    def __init__(
        self,
        stateCount,
        actionCount,
        stepSize,
        discount,
        epsilon,
        beta,
        c,
        decode_func,
    ):
        self.beta = beta
        self.c = c
        self.decode = decode_func
        super().__init__(stateCount, actionCount, stepSize, discount, epsilon)

    def _getKMatrix(self, state):
        row, col, pLoc, dLoc = self.decode(state)
        kMatrix = 1 / (
            1 + np.exp(self.beta * (self._get_manhattan_matrix((row, col)) - self.c))
        )

        rows = np.arange(5).reshape(5, 1) * 100
        cols = np.arange(5) * 20
        gridBase = rows + cols

        if pLoc < 4:
            offsets = (pLoc * 4) + np.arange(4)
            offsets = offsets.reshape(-1, 1, 1)
            stateMatrix = gridBase + offsets
        else:
            offset = (pLoc * 4) + dLoc
            stateMatrix = gridBase + offset

        return kMatrix, stateMatrix

    def update(self, state, action, reward, next_state, next_action, terminated):
        td = self._TDError(state, action, reward, next_state, next_action, terminated)
        super().update(state, action, reward, next_state, next_action, terminated)

        kMatrix, stateMatrix = self._getKMatrix(state)
        self.qMatrix[stateMatrix, action] += self.stepSize * kMatrix * td

    def _get_manhattan_matrix(self, center):
        x, y = np.ogrid[0:5, 0:5]
        distance = np.abs(x - center[0]) + np.abs(y - center[1])
        return distance
