# Quick Start

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Code

```bash
python main.py
```

This runs the full pipeline and creates all graphs and animations.

Alternatively, you can run each of the following components separately:

- **Train agents and create performance metrics data and graphs:**

```bash
python metrics_test.py; python draw_metrics.py
```

- **Run the hyperparameter sweeps and create the sweep graphs:**

```bash
python sweep_test.py
```

- **Create the simulation video and GIF files:**

```bash
python create_animation.py
```

## Outputs

For all previous commands, the results will be available in the `results/` folder.

---

## Project Structure

```
RLProject/
├── .gitignore                          # Ignored files (.venv, output/, results/)
├── requirements.txt                    # Project dependencies
├── main.py                             # Main pipeline execution entry point
├── helpers.py                          # Multi-seed agent training loops
├── metrics_wrapper.py                  # Gym wrapper tracking success rate and length
├── metrics_test.py                     # Runs agent/reward benchmarks (saves CSVs)
├── draw_metrics.py                     # Plots reward and algorithm comparisons
├── sweep_test.py                       # Runs hyperparameter sweeps and plots
├── create_animation.py                 # Generates gameplay videos and GIFs
├── algorithms/                         # RL agent variants
│   ├── __init__.py                     # Package imports
│   ├── SARSA.py                        # Standard tabular SARSA
│   ├── SARSAGen.py                     # SARSA with spatial distance generalization
│   ├── PSSARSA.py                      # Prioritized Sweeping SARSA
│   └── PSSARSAGen.py                   # Prioritized Sweeping + generalization
├── rewards/                            # Custom reward configurations
│   ├── __init__.py                     # Package imports
│   ├── Sparse.py                       # Sparse rewards wrapper (+1 on success)
│   ├── Custom1.py                      # Basic shaped rewards wrapper
│   └── Custom2.py                      # Advanced loop-penalizing rewards wrapper
├── output/                             # Raw CSV data logs (Created at runtime)
│   ├── agents/                         # Agent checkpoints
│   ├── metrics/                        # Benchmark CSV files
│   └── sweep/                          # Hyperparameter sweep CSV files
└── results/                            # Generated plots and media (Created at runtime)
    ├── compare_algorithm/              # Algorithm comparison plots
    ├── compare_reward/                 # Reward function comparison plots
    ├── sweep/                          # Hyperparameter sweep plots
    └── video/                          # Gameplay MP4 and GIF files
```
