from gymnasium import Wrapper


class Custom2(Wrapper):
    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self.curState = state
        self.prevState = -1
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if next_state == self.curState:  # illegal move or stuck in a loop
            reward = -10
        elif next_state == self.prevState:
            reward = -10
        elif action == 4 and reward == -10:  # illegal pickup
            reward = -10
        elif action == 5 and reward == -10:  # illegal dropoff
            reward = -20
        elif reward == 20:  # sucessfull dropoff
            reward = 50
        elif reward == -1:  # every move
            reward = -5
        else:
            reward = 0

        self.prevState = self.curState
        self.curState = next_state
        return next_state, reward, terminated, truncated, info
