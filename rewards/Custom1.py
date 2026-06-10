from gymnasium import Wrapper


class Custom1(Wrapper):
    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self.curState = state
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if action == 4 and reward == -10:  # illegal pickup
            reward = -1
        elif action == 5 and reward == -10:  # illegal dropoff
            reward = -10
        elif reward == 20:  # sucessfull dropoff
            reward = 20
        elif reward == -1:  # every move
            reward = -1
        else:
            reward = 0

        self.curState = next_state
        return next_state, reward, terminated, truncated, info
