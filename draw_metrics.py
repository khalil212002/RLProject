import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.makedirs("results", exist_ok=True)
os.makedirs("results/compare_reward", exist_ok=True)
os.makedirs("results/compare_algorithm", exist_ok=True)
os.makedirs("output/metrics", exist_ok=True)


def draw_from_csv_compare_reward(
    algorithms=[
        "sarsa",
        "sarsa+priority_sweep",
        "sarsa+generalization",
        "sarsa+priority_sweep+generalization",
    ]
):
    print("\n\n-----Starting reward compare drawing------\n")
    rewards = ["default", "sparse", "custom1", "custom2"]

    sb.set_theme(style="darkgrid")

    for al in tqdm(algorithms, desc="Processing algorithms"):
        algo_df_list = []
        tqdm.write(f"drawing {al} reward comparison.")

        for r in rewards:
            file_path = f"output/metrics/{al}_metrics_{r}_reward.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Reward Design"] = r.capitalize()
                df["Success Rate Smoothed"] = df.groupby("Seed")[
                    "Success Rate"
                ].transform(lambda x: x.rolling(window=25, min_periods=1).mean())
                df["Length Smoothed"] = df.groupby("Seed")["Episode Length"].transform(
                    lambda x: x.rolling(window=25, min_periods=1).mean()
                )

                algo_df_list.append(df)
            else:
                print(f"Skipping: {file_path} not found.")

        if algo_df_list:
            combined_df = pd.concat(algo_df_list, axis=0, ignore_index=True)
            plt.figure(figsize=(10, 6))
            sb.lineplot(
                data=combined_df,
                x="Episode",
                y="Success Rate",
                hue="Reward Design",
                errorbar=(
                    "ci",
                    95,
                ),
            )

            plt.title(f"Success Rate Comparison - {al.capitalize()} Algorithm")
            plt.xlabel("Episode")
            plt.ylabel("Success Rate (Rolling Mean = 25)")
            plt.legend(title="Reward Function", loc="lower right")
            plt.savefig(
                f"results/compare_reward/{al}_success_rate.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(10, 6))
            sb.lineplot(
                data=combined_df,
                x="Episode",
                y="Episode Length",
                hue="Reward Design",
                errorbar=("ci", 95),
            )

            plt.title(f"Episode Length Comparison - {al.capitalize()} Algorithm")
            plt.xlabel("Episode")
            plt.ylabel("Episode Length (Rolling Mean = 25)")

            plt.legend(title="Reward Function", loc="upper right")

            plt.savefig(
                f"results/compare_reward/{al}_length.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()


def draw_from_csv_compare_algorithm(
    rewards=["default", "sparse", "custom1", "custom2"]
):
    print("\n\n-----Starting algorithm compare drawing------\n")
    algos = [
        "sarsa",
        "sarsa+priority_sweep",
        "sarsa+generalization",
        "sarsa+priority_sweep+generalization",
    ]

    sb.set_theme(style="darkgrid")

    for r in tqdm(rewards, desc="Processing rewards"):
        tqdm.write(f"drawing {r} algorithm comparison")
        reward_df_list = []

        for a in algos:
            file_path = f"output/metrics/{a}_metrics_{r}_reward.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["algo"] = a
                reward_df_list.append(df)
            else:
                print(f"Skipping: {file_path} not found.")

        if reward_df_list:
            combined_df = pd.concat(reward_df_list, axis=0, ignore_index=True)

            plt.figure(figsize=(10, 6))

            sb.lineplot(
                data=combined_df,
                x="Episode",
                y="Success Rate",
                hue="algo",
                errorbar=("ci", 95),
            )

            plt.title(f"Success Rate Comparison - {r.capitalize()} Reward")
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")

            plt.savefig(
                f"results/compare_algorithm/{r}_reward_success_rate.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(10, 6))

            sb.lineplot(
                data=combined_df,
                x="Episode",
                y="Episode Length",
                hue="algo",
                errorbar=("ci", 95),
            )

            plt.title(f"Episode Length Comparison - {r.capitalize()} Reward")
            plt.xlabel("Episode")
            plt.ylabel("Episode Length")

            plt.savefig(
                f"results/compare_algorithm/{r}_reward_length.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    draw_from_csv_compare_reward()
    draw_from_csv_compare_algorithm()
