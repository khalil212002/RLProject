from gymnasium import RewardWrapper


class Sparse(RewardWrapper):
    def reward(self, reward):
        if self.reward == 20:  # successful dropoff
            return 1
        return 0  # everything else
