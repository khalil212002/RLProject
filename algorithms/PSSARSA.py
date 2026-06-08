from algorithms.SARSA import SARSA
import numpy as np
import heapq


class PSSARSA(SARSA):
    def __init__(
        self, stateCount, actionCount, stepSize, discount, epsilon, threshold, steps
    ):
        super().__init__(stateCount, actionCount, stepSize, discount, epsilon)
        self.threshold = threshold
        self.steps = steps
        self.model = {}
        self.pQueue = []
        self.predecessors = {}

    def update(self, state, action, reward, next_state, next_action, terminated):
        self.model[(state, action)] = (reward, next_state, terminated)
        pred = self.predecessors.get(next_state, None)
        if pred is None:
            self.predecessors[next_state] = set()
            pred = self.predecessors[next_state]
        pred.add((state, action))
        p = abs(
            super()._TDError(state, action, reward, next_state, next_action, terminated)
        )
        if p > self.threshold:
            heapq.heappush(self.pQueue, (-p, state, action))

        for step in range(self.steps):
            if len(self.pQueue) == 0:
                break
            p, s, a = heapq.heappop(self.pQueue)
            r, ns, term = self.model[(s, a)]
            na = self.choose_action(ns)
            super().update(s, a, r, ns, na, term)
            for ps, pa in self.predecessors.get(s, []):
                pr, pns, pterm = self.model[(ps, pa)]
                pna = self.choose_action(pns)
                pp = abs(super()._TDError(ps, pa, pr, pns, pna, pterm))
                if pp > self.threshold:
                    heapq.heappush(self.pQueue, (-pp, ps, pa))
