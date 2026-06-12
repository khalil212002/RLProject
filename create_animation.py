from helpers import train
from algorithms import PSSARSAGen
from rewards import *
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def main():
    env = gym.make("Taxi-v4", is_rainy=False, render_mode="rgb_array")
    envs = {}
    envs["default"] = env
    envs["sparse"] = Sparse(env)
    envs["custom1"] = Custom1(env)
    envs["custom2"] = Custom2(env)

    for en in envs.keys():
        curEnv = RecordVideo(
            envs[en],
            video_folder="results/video",
            name_prefix=en,
            episode_trigger=lambda x: (x + 1) % 1000 == 0,
        )
        agent = PSSARSAGen(
            curEnv.observation_space.n,
            curEnv.action_space.n,
            0.25,
            0.99,
            0.1,
            0.2,
            10,
            4,
            0.5,
            curEnv.unwrapped.decode,
        )
        train(agent, curEnv, 1000, 1)
        curEnv.close()


if __name__ == "__main__":
    main()
