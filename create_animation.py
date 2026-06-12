from helpers import train
from algorithms import PSSARSAGen
from rewards import *
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

os.makedirs("results", exist_ok=True)
os.makedirs("results/video", exist_ok=True)

# moviepy library already used by the wrapper we only use it to convert vid to gif
from moviepy import VideoFileClip


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

        try:
            vid = VideoFileClip(f"results/video/{en}-episode-999.mp4")
            vid.write_gif(f"results/video/{en}-episode-999.gif")
        except:
            pass


if __name__ == "__main__":
    main()
