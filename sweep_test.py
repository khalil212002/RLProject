import gymnasium as gym
from metrics_test import getMetricsOver5Seeds
from algorithms import PSSARSAGen
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import tqdm
from rewards import Custom2
import os

os.makedirs("output", exist_ok=True)
os.makedirs("output/sweep", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/sweep", exist_ok=True)


def draw_sweep_from_csv(
    sweeps=["planning", "discount"],
    rewards=["base", "custom2"],
    graphs=["Success Rate Smoothed", "Length Smoothed"],
):
    print("\n\n-----Starting sweep drawing------\n")
    for s in tqdm.tqdm(sweeps, desc="Parameters"):
        for r in tqdm.tqdm(rewards, desc="Rewards"):
            tqdm.tqdm.write(f"\n-- drawing {r} reward, sweeping {s} --")

            file_path = f"output/sweep/{s}_sweep_{r}_reward.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Success Rate Smoothed"] = df.groupby("Seed")[
                    "Success Rate"
                ].transform(lambda x: x.rolling(window=25, min_periods=1).mean())
                df["Length Smoothed"] = df.groupby("Seed")["Episode Length"].transform(
                    lambda x: x.rolling(window=25, min_periods=1).mean()
                )
            else:
                tqdm.tqdm.write(f"Skipping: {file_path} not found.")
                continue

            for g in graphs:
                tqdm.tqdm.write(f"Drawing {g}.")
                plt.figure(figsize=(10, 6))
                sb.set_theme(style="darkgrid")

                sb.lineplot(
                    data=df,
                    x="Episode",
                    y=g,
                    hue=s,
                    errorbar=("ci", 95),
                )

                metric_label = "Success Rate" if "Success" in g else "Episode Length"
                plt.title(
                    f"Hyperparameter Sweep: {s.capitalize()} (Environment: {r.capitalize()} Reward)"
                )
                plt.xlabel("Episode")
                plt.ylabel(f"{metric_label} (Rolling Mean = 25)")
                plt.legend(title=s.capitalize(), loc="best")

                file_suffix = "success_rate" if "Success" in g else "episode_length"
                plt.savefig(
                    f"results/sweep/{s}_sweep_{r}_{file_suffix}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()


def sweepTest(episodes=1000, rewards=["base", "custom2"], planning=True, discount=True):
    tqdm.tqdm.write("\n\n-----Starting sweep test------\n")

    if planning:
        tqdm.tqdm.write("---Sweeping For planning---\n")
        for r in rewards:
            env = gym.make("Taxi-v4", is_rainy=False)
            if r == "custom2":
                env = Custom2(env)
            df = []
            for d in tqdm.tqdm([1, 5, 10, 15], desc=f"Planning steps ({r})"):
                tqdm.tqdm.write(f"\n-- {r} reward, Planning = {d} --")
                agent = PSSARSAGen(
                    env.observation_space.n,
                    env.action_space.n,
                    0.25,
                    0.99,
                    0.1,
                    0.2,
                    d,
                    4,
                    0.5,
                    env.unwrapped.decode,
                )

                ndf = getMetricsOver5Seeds(agent, env, episodes, [1, 2, 3, 4, 5])
                ndf["planning"] = d
                df.append(ndf)

            df = pd.concat(df, axis=0, ignore_index=True)
            df.to_csv(f"output/sweep/planning_sweep_{r}_reward.csv", index=False)
            env.close()

    if discount:
        tqdm.tqdm.write("\n\n---Sweeping For discount---\n")
        for r in ["base", "custom2"]:
            env = gym.make("Taxi-v4", is_rainy=False)
            if r == "custom2":
                env = Custom2(env)
            df = []
            for d in tqdm.tqdm([0.3, 0.5, 0.7, 0.9], desc=f"Discount factor ({r})"):
                tqdm.tqdm.write(f"\n-- {r} reward, Discount = {d} --")
                agent = PSSARSAGen(
                    env.observation_space.n,
                    env.action_space.n,
                    0.25,
                    d,
                    0.1,
                    0.2,
                    10,
                    4,
                    0.5,
                    env.unwrapped.decode,
                )

                ndf = getMetricsOver5Seeds(agent, env, episodes, [1, 2, 3, 4, 5])
                ndf["discount"] = d
                df.append(ndf)

            df = pd.concat(df, axis=0, ignore_index=True)
            df.to_csv(f"output/sweep/discount_sweep_{r}_reward.csv", index=False)
            env.close()


if __name__ == "__main__":
    sweepTest()
    draw_sweep_from_csv()
