from gymnasium import Wrapper


class MetricsWrapper(Wrapper):
    def __init__(self, env, successList, lengthsList):
        super().__init__(env)
        self.successRate = successList
        self.avgLength = lengthsList
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        self.t += 1
        if terminated or truncated:
            self.successRate.append(1 if terminated else 0)
            self.avgLength.append(self.t)
        return next_state, reward, terminated, truncated, info
