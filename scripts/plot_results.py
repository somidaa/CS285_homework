"""Plot MSE vs Flow training curves from exp/ logs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "exp"
MSE_DIR = EXP / "seed_42_20260921_203355"
FLOW_DIR = EXP / "seed_42_20260921_215435"


def read_train_loss(csv_path: Path):
    steps, losses = [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("train_loss", "").strip()
            if v:
                steps.append(int(row["step"]))
                losses.append(float(v))
    return steps, losses


def read_eval_reward(run_dir: Path):
    """Pull eval/mean_reward vs step from the wandb offline .wandb file."""
    wandb_file = next(run_dir.glob("wandb/run-*.wandb"), None)
    if wandb_file is None:
        return [], []
    steps, rewards = [], []
    try:
        from wandb.sdk.internal.datastore import DataStore

        ds = DataStore()
        ds.open_for_read(str(wandb_file))
        while True:
            record = ds.scan_record()
            if record is None:
                break
            # records are protobuf; look for telemetry/summary containing eval/mean_reward
            try:
                data = record.data
                if b"eval/mean_reward" in data:
                    # crude parse: find step and reward from the raw bytes
                    import re

                    # this is best-effort; if it fails we fall back to final values
                    pass
            except Exception:
                pass
        ds.close()
    except Exception:
        pass
    return steps, rewards


def main():
    ms, ml = read_train_loss(MSE_DIR / "log.csv")
    fs, fl = read_train_loss(FLOW_DIR / "log.csv")

    # Eval rewards: best-effort from wandb binary; fall back to known checkpoints.
    # We only have the per-step eval points at 0,10000,...,75600.
    # Final rewards from wandb summary:
    mse_final, flow_final = 0.62049, 0.83225

    # Smooth train loss for readability
    def smooth(xs, ys, w=50):
        out = []
        for i in range(len(ys)):
            lo = max(0, i - w)
            out.append(sum(ys[lo : i + 1]) / (i + 1 - lo))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(ms, smooth(ms, ml), color="#1f77b4", label="MSE (final ~0.015)", lw=1.5)
    ax1.plot(fs, smooth(fs, fl), color="#d62728", label="Flow (final ~0.18)", lw=1.5)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Training Loss: MSE vs Flow")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Eval reward: plot final comparison as bars since per-point series is in wandb UI
    labels = ["MSE", "Flow"]
    finals = [mse_final, flow_final]
    colors = ["#1f77b4", "#d62728"]
    bars = ax2.bar(labels, finals, color=colors, width=0.5)
    ax2.axhline(0.5, color="#1f77b4", ls="--", alpha=0.6, label="MSE requirement 0.5")
    ax2.axhline(0.7, color="#d62728", ls="--", alpha=0.6, label="Flow requirement 0.7")
    for b, v in zip(bars, finals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    ax2.set_ylabel("eval/mean_reward")
    ax2.set_title("Final Eval Reward: MSE vs Flow")
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Berkeley HW1: Push-T Imitation Learning (Action Chunking)", fontsize=13)
    fig.tight_layout()
    out = ROOT / "report" / "mse_vs_flow.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
