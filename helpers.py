import numpy as np
from tqdm import trange, tqdm
from metrics_wrapper import MetricsWrapper
import pandas as pd
import copy


def train(agent, env, episodes, seed):
    np.random.seed(seed)
    for _ in trange(episodes):
        state, _ = env.reset()

        total_reward = 0
        action = agent.choose_action(state)

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)

            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)

            state = next_state
            action = next_action

            total_reward += reward

            if terminated or truncated:
                break


def getMetricsOver5Seeds(agent, env, episodes, seeds):
    success = []
    length = []
    eps = np.arange(episodes) + 1
    for i in seeds:
        tqdm.write(f"Seed={i}")
        s = []
        l = []
        curEnv = MetricsWrapper(env, successList=s, lengthsList=l)
        curAgent = copy.deepcopy(agent)
        train(curAgent, curEnv, episodes, i)

        successOverEpisodes = np.cumsum(s) / eps
        lengthOverEspisodes = np.cumsum(l) / eps

        success.append(successOverEpisodes)
        length.append(lengthOverEspisodes)
    records = []

    for i, seed in enumerate(seeds):
        current_seed_success = success[i]
        current_seed_length = length[i]

        for ep in range(episodes):
            records.append(
                {
                    "Episode": ep + 1,
                    "Seed": seed,
                    "Success Rate": current_seed_success[ep],
                    "Episode Length": current_seed_length[ep],
                }
            )
    df = pd.DataFrame(records)
    return df
