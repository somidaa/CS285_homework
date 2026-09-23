"""Plot all HW2 experiment results."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "exp"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)


def read_log(path):
    steps, returns = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = row.get("Eval_AverageReturn", "").strip()
            step = row.get("Train_EnvstepsSoFar", "").strip()
            if ev and step:
                try:
                    returns.append(float(ev))
                    steps.append(float(step))
                except ValueError:
                    pass
    return steps, returns


def plot_group(experiments, title, filename, ylabel="Eval_AverageReturn"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, dirname, color in experiments:
        d = EXP / dirname
        csv_path = d / "log.csv"
        if not csv_path.exists():
            print(f"missing {csv_path}")
            continue
        s, r = read_log(csv_path)
        ax.plot(s, r, label=name, color=color, lw=1.5)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    print(f"saved {filename}")
    plt.close()


# Q1: CartPole batch/RTG comparison
plot_group([
    ("small batch, no RTG, no norm", "CartPole-v0_q1_sb_sd1_20260923_144348", "#d62728"),
    ("small batch, RTG+norm", "CartPole-v0_q1_sb_rtgna_sd1_20260923_143609", "#1f77b4"),
    ("large batch, no RTG, no norm", "CartPole-v0_q1_lb_sd1_20260923_144543", "#ff7f0e"),
    ("large batch, RTG+norm", "CartPole-v0_q1_lb_rtgna_sd1_20260923_144943", "#2ca02c"),
], "Q1: CartPole - Batch Size & RTG Comparison", "q1_cartpole.png")

# Q2: HalfCheetah baseline
plot_group([
    ("no baseline", "HalfCheetah-v4_q2_no_baseline_sd1_20260923_161329", "#d62728"),
    ("with baseline", "HalfCheetah-v4_q2_baseline_sd1_20260923_161851", "#1f77b4"),
], "Q2: HalfCheetah - Baseline Comparison", "q2_halfcheetah.png")

# Q3: LunarLander GAE
plot_group([
    ("gae_lambda=0", "LunarLander-v2_q3_gae0_sd1_20260923_164134", "#d62728"),
    ("gae_lambda=0.95", "LunarLander-v2_q3_gae095_sd1_20260923_165407", "#ff7f0e"),
    ("gae_lambda=0.98", "LunarLander-v2_q3_gae098_sd1_20260923_182655", "#2ca02c"),
    ("gae_lambda=0.99", "LunarLander-v2_q3_gae099_sd1_20260923_183224", "#1f77b4"),
    ("gae_lambda=1.0", "LunarLander-v2_q3_gae1_sd1_20260923_183704", "#9467bd"),
], "Q3: LunarLander - GAE Lambda Sweep", "q3_lunarlander_gae.png")

# Q4: InvertedPendulum
plot_group([
    ("baseline config", "InvertedPendulum-v4_q4_baseline_sd1_20260923_163421", "#1f77b4"),
], "Q4: InvertedPendulum (target 1000)", "q4_invertedpendulum.png")
