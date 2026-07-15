"""Plotting utilities for the per-checkpoint train/test loss, R^2, and MAE curves
written to comparison_log.csv by Fourier-MIONet_sg.py.

generate_predictions_and_plots.py calls these directly against a real
comparison_log.csv; the __main__ block below only exercises them against synthetic
data as a smoke test.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_comparison_log(csv_path):
    return pd.read_csv(csv_path)


def plot_loss_curves(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["step"], df["fourier_train_loss"], label="Fourier-MIONet train", color="tab:orange", linestyle="--")
    ax.plot(df["step"], df["fourier_test_loss"], label="Fourier-MIONet test", color="tab:orange")
    ax.plot(df["step"], df["wno_train_loss"], label="U-WNO-MIONet train", color="tab:blue", linestyle="--")
    ax.plot(df["step"], df["wno_test_loss"], label="U-WNO-MIONet test", color="tab:blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean L2 relative error")
    ax.set_title("Train / test loss vs. training step")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_r2_curves(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["step"], df["fourier_test_r2_plume"], label="Fourier-MIONet", color="tab:orange")
    ax.plot(df["step"], df["wno_test_r2_plume"], label="U-WNO-MIONet", color="tab:blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test R$^2$ (plume region)")
    ax.set_title("Plume-region R$^2$ vs. training step")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mae_curves(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["step"], df["fourier_test_mae_plume"], label="Fourier-MIONet", color="tab:orange")
    ax.plot(df["step"], df["wno_test_mae_plume"], label="U-WNO-MIONet", color="tab:blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test MAE (plume region)")
    ax.set_title("Plume-region MAE vs. training step")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_comparisons(csv_path, out_dir):
    """Load `csv_path` (may be partial, e.g. from an interrupted run) and write all
    three curve plots into `out_dir`. Returns the list of paths written."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_comparison_log(csv_path)
    return [
        plot_loss_curves(df, os.path.join(out_dir, "loss_curves.png")),
        plot_r2_curves(df, os.path.join(out_dir, "r2_curves.png")),
        plot_mae_curves(df, os.path.join(out_dir, "mae_curves.png")),
    ]


if __name__ == "__main__":
    # Synthetic-data smoke test — exercises every function above without needing a
    # real comparison_log.csv. generate_predictions_and_plots.py calls
    # plot_all_comparisons() directly against the real CSV instead of this block.
    import tempfile

    import numpy as np

    rng = np.random.RandomState(0)
    steps = np.arange(0, 500, 10)
    df = pd.DataFrame({
        "step": steps,
        "fourier_train_loss": 1.0 * np.exp(-steps / 300) + rng.normal(0, 0.02, len(steps)),
        "fourier_test_loss": 1.0 * np.exp(-steps / 300) + rng.normal(0, 0.03, len(steps)),
        "fourier_test_r2_plume": 1 - np.exp(-steps / 300) + rng.normal(0, 0.02, len(steps)),
        "fourier_test_mae_plume": 0.2 * np.exp(-steps / 300) + rng.normal(0, 0.01, len(steps)),
        "wno_train_loss": 0.9 * np.exp(-steps / 250) + rng.normal(0, 0.02, len(steps)),
        "wno_test_loss": 0.9 * np.exp(-steps / 250) + rng.normal(0, 0.03, len(steps)),
        "wno_test_r2_plume": 1 - np.exp(-steps / 250) + rng.normal(0, 0.02, len(steps)),
        "wno_test_mae_plume": 0.18 * np.exp(-steps / 250) + rng.normal(0, 0.01, len(steps)),
    })
    out_dir = tempfile.mkdtemp(prefix="plot_comparison_demo_")
    csv_path = os.path.join(out_dir, "demo_comparison_log.csv")
    df.to_csv(csv_path, index=False)
    for p in plot_all_comparisons(csv_path, out_dir):
        print("wrote", p)
