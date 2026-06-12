from algorithms import SARSA, PSSARSA, SARSAGen, PSSARSAGen
from helpers import getMetricsOver5Seeds
import gymnasium as gym
from rewards import Sparse, Custom1, Custom2
import os
import tqdm

os.makedirs("output", exist_ok=True)
os.makedirs("output/metrics", exist_ok=True)
os.makedirs("output/agents", exist_ok=True)


def testSparse(episodes=1000):
    tqdm.tqdm.write("--Starting Sparse drawing--\n")
    env = Sparse(gym.make("Taxi-v4", is_rainy=False))
    agents = {}
    agents["sarsa"] = SARSA(
        env.observation_space.n, env.action_space.n, 5000, 0.99, 0.8
    )
    agents["sarsa+priority_sweep"] = PSSARSA(
        env.observation_space.n, env.action_space.n, 5000, 0.99, 0.8, 0.01, 10
    )
    agents["sarsa+generalization"] = SARSAGen(
        env.observation_space.n,
        env.action_space.n,
        5000,
        0.99,
        0.8,
        4,
        0.5,
        env.unwrapped.decode,
    )
    agents["sarsa+priority_sweep+generalization"] = PSSARSAGen(
        env.observation_space.n,
        env.action_space.n,
        5000,
        0.99,
        0.8,
        0.01,
        10,
        4,
        0.5,
        env.unwrapped.decode,
    )

    for ag in tqdm.tqdm(agents.keys()):
        tqdm.tqdm.write(f"staring {ag} training with sparse rewards")

        df = getMetricsOver5Seeds(agents[ag], env, episodes, [1, 2, 3, 4, 5])
        df.to_csv(f"output/metrics/{ag}_metrics_sparse_reward.csv", index=False)


def metricsTest(episodes=1000):
    print("\n\n----This is metrics test for all agents with all rewards:----\n")

    env = gym.make("Taxi-v4", is_rainy=False)
    envs = {}
    envs["default"] = env
    envs["sparse"] = Sparse(env)
    envs["custom1"] = Custom1(env)
    envs["custom2"] = Custom2(env)

    agents = {}
    agents["sarsa"] = SARSA(
        env.observation_space.n, env.action_space.n, 0.25, 0.99, 0.1
    )
    agents["sarsa+priority_sweep"] = PSSARSA(
        env.observation_space.n, env.action_space.n, 0.25, 0.99, 0.1, 0.2, 10
    )
    agents["sarsa+generalization"] = SARSAGen(
        env.observation_space.n,
        env.action_space.n,
        0.25,
        0.99,
        0.1,
        4,
        0.5,
        env.unwrapped.decode,
    )
    agents["sarsa+priority_sweep+generalization"] = PSSARSAGen(
        env.observation_space.n,
        env.action_space.n,
        0.25,
        0.99,
        0.1,
        0.2,
        10,
        4,
        0.5,
        env.unwrapped.decode,
    )

    for en in tqdm.tqdm(envs.keys()):
        if en == "sparse":
            testSparse(episodes)
            continue
        for ag in tqdm.tqdm(agents.keys()):
            tqdm.tqdm.write(f"\n\nstaring {ag} training with {en} rewards")

            df = getMetricsOver5Seeds(agents[ag], envs[en], episodes, [1, 2, 3, 4, 5])
            df.to_csv(f"output/metrics/{ag}_metrics_{en}_reward.csv", index=False)


if __name__ == "__main__":
    metricsTest()
