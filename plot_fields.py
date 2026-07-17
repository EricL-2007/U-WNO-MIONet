"""Plotting utilities for MIONet-style field predictions on a
(24 timesteps) x (96 depth) x (200 radial) grid.

Every function takes real arrays and an explicit output path, and returns that path.
generate_predictions_and_plots.py calls these directly with real prediction arrays;
the __main__ block below only exercises them against synthetic data as a smoke test,
so this file can be sanity-checked (`python plot_fields.py`) without any checkpoints
or data files on disk.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def build_plume_mask(y_true):
    """Boolean mask, shape (N, 24, 96, 200), True where a cell is inside the CO2 plume.

    Mirrors the masking convention used for the R2_plume/MAE_plume training metrics in
    Fourier-UWNO-MIONet_sg.py: a depth row is excluded only if its near-well column (r=0) is
    still ~0 at the final timestep, and that exclusion is broadcast across all timesteps
    and radial positions for that (sample, z) pair. Keeping the same convention here
    means the plots' R2/MAE line up with the numbers already logged during training.
    """
    z_axis = y_true[:, -1, :, 0]  # (N, 96)
    row_mask = ~np.isclose(z_axis, 0.0)  # (N, 96)
    return np.broadcast_to(row_mask[:, None, :, None], y_true.shape)


def _shared_vmin_vmax(arrays):
    vmin = min(float(np.nanmin(a)) for a in arrays)
    vmax = max(float(np.nanmax(a)) for a in arrays)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_field_snapshots(y_true, y_preds, case_idx, time_indices, out_path,
                          field_name="Gas Saturation", time_labels=None, cmap="viridis"):
    """Grid of (row = True + each model, col = selected timesteps) field maps for one
    test case. `y_preds` is {model_name: array shaped like y_true}."""
    if time_labels is None:
        time_labels = [f"step {t}" for t in time_indices]

    rows = [("True", y_true)] + list(y_preds.items())
    frames = [arr[case_idx, t] for _, arr in rows for t in time_indices]
    vmin, vmax = _shared_vmin_vmax(frames)

    fig, axes = plt.subplots(
        len(rows), len(time_indices),
        figsize=(3.2 * len(time_indices), 3.0 * len(rows)),
        squeeze=False,
    )
    im = None
    for r, (name, arr) in enumerate(rows):
        for c, t in enumerate(time_indices):
            ax = axes[r][c]
            im = ax.imshow(arr[case_idx, t], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if r == 0:
                ax.set_title(time_labels[c])
            if c == 0:
                ax.set_ylabel(name)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{field_name} — test case {case_idx}")
    fig.colorbar(im, ax=axes, shrink=0.8, label=field_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_radial_profile(y_true, y_preds, case_idx, z_index, time_index, out_path,
                         field_name="Gas Saturation"):
    """Line plot of field value vs. radial index, at a fixed depth/time, for one case."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_true[case_idx, time_index, z_index, :], label="True", color="black", linewidth=2)
    for name, arr in y_preds.items():
        ax.plot(arr[case_idx, time_index, z_index, :], label=name, linestyle="--")
    ax.set_xlabel("Radial index r")
    ax.set_ylabel(field_name)
    ax.set_title(f"Radial profile — case {case_idx}, z={z_index}, t={time_index}")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scatter_true_vs_pred(y_true, y_pred, out_path, model_name,
                               field_name="Gas Saturation", mask=None,
                               max_points=50000, seed=0):
    """Scatter of true vs. predicted values, optionally restricted to `mask` (boolean,
    same shape as y_true/y_pred)."""
    yt = np.asarray(y_true[mask] if mask is not None else y_true).reshape(-1)
    yp = np.asarray(y_pred[mask] if mask is not None else y_pred).reshape(-1)

    if yt.size > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(yt.size, size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]

    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=2, alpha=0.15, color="tab:blue")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="y = x")
    ax.set_xlabel(f"True {field_name}")
    ax.set_ylabel(f"Predicted {field_name}")
    ax.set_title(f"{model_name}: true vs. predicted")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_r2_histogram(y_true, y_pred, out_path, model_name, field_name="Gas Saturation", mask=None):
    """Histogram of per-test-case R^2, optionally restricted to `mask`."""
    n = y_true.shape[0]
    r2_values = []
    for i in range(n):
        if mask is not None:
            yt = y_true[i][mask[i]].flatten()
            yp = y_pred[i][mask[i]].flatten()
        else:
            yt = y_true[i].flatten()
            yp = y_pred[i].flatten()
        if yt.size == 0:
            continue
        sst = np.sum(np.square(yt - np.mean(yt)))
        if sst == 0:
            continue
        sse = np.sum(np.square(yt - yp))
        r2_values.append(1 - sse / sst)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r2_values, bins=min(20, max(5, len(r2_values))), color="tab:blue", edgecolor="black")
    if r2_values:
        ax.axvline(float(np.mean(r2_values)), color="red", linestyle="--",
                    label=f"mean = {np.mean(r2_values):.3f}")
        ax.legend()
    ax.set_xlabel(f"Per-case R$^2$ ({field_name})")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: per-case R$^2$ distribution (n={len(r2_values)})")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mean_error_map(y_true, y_pred, out_path, model_name, field_name="Gas Saturation"):
    """Spatial map of mean signed error (pred - true), averaged over samples and time."""
    error = np.mean(y_pred - y_true, axis=(0, 1))  # (96, 200)
    vmax = float(np.max(np.abs(error)))
    vmax = vmax if vmax > 0 else 1e-6
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label=f"Mean error ({field_name})")
    ax.set_title(f"{model_name}: mean signed error map")
    ax.set_xlabel("Radial index r")
    ax.set_ylabel("Depth index z")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_input_context(permeability, case_idx, out_path):
    """Map of the input field (e.g. permeability) for one test case, shape (N, 96, 200)."""
    field = permeability[case_idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.all(field > 0):
        norm = mcolors.LogNorm(vmin=field.min(), vmax=field.max())
        im = ax.imshow(field, cmap="cividis", norm=norm, aspect="auto")
    else:
        im = ax.imshow(field, cmap="cividis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Input field (channel 0)")
    ax.set_title(f"Input context — test case {case_idx}")
    ax.set_xlabel("Radial index r")
    ax.set_ylabel("Depth index z")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Synthetic-data smoke test — exercises every function above without needing real
    # checkpoints or data files. generate_predictions_and_plots.py calls the functions
    # directly with real arrays instead of this block.
    import tempfile

    rng = np.random.RandomState(0)
    n, t, z, r = 3, 24, 96, 200
    y_true = np.clip(rng.rand(n, t, z, r) - 0.5, 0, None)
    y_pred_a = y_true + rng.normal(0, 0.02, size=y_true.shape)
    y_pred_b = y_true + rng.normal(0, 0.05, size=y_true.shape)
    permeability = np.exp(rng.normal(2, 3, size=(n, z, r)))

    out_dir = tempfile.mkdtemp(prefix="plot_fields_demo_")
    mask = build_plume_mask(y_true)
    preds = {"ModelA": y_pred_a, "ModelB": y_pred_b}

    plot_field_snapshots(y_true, preds, 0, [2, 8, 23], os.path.join(out_dir, "snapshots.png"))
    plot_radial_profile(y_true, preds, 0, z // 2, 18, os.path.join(out_dir, "radial.png"))
    plot_scatter_true_vs_pred(y_true, y_pred_a, os.path.join(out_dir, "scatter.png"), "ModelA", mask=mask)
    plot_r2_histogram(y_true, y_pred_a, os.path.join(out_dir, "r2_hist.png"), "ModelA", mask=mask)
    plot_mean_error_map(y_true, y_pred_a, os.path.join(out_dir, "error_map.png"), "ModelA")
    plot_input_context(permeability, 0, os.path.join(out_dir, "input_context.png"))
    print(f"Demo plots written to {out_dir}")
